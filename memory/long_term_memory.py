"""
长期记忆模块
核心能力：跨会话持久化核心事实、用户偏好、关键结论，支持语义召回
工业级特性：分级写入策略、相似度阈值过滤、失败自动降级、批量缓冲写入
设计原则：单一职责——只做记忆业务逻辑，向量存取完全交给VectorStore
"""

import hashlib
import os
from typing import List, Dict, Optional
from memory.base_memory import BaseMemory
from memory.memory_policy import MemoryPolicyMixin
from rag_knowledge.vector_store import VectorStore
from client.embedding_client import EmbeddingClient
from pydantic_schemas.memory_policy_schema import MemoryLevel, MemoryPolicyConfig
from utils.logger_tool import logger
from utils.path_tool import get_abs_path

class LongTermMemory(BaseMemory, MemoryPolicyMixin):
    """
    长期记忆类
    继承BaseMemory：统一接口，agent可无缝替换记忆类型
    继承MemoryPolicyMixin：复用敏感检测、去重、分级打标能力
    """
    def __init__(
            self,
            policy_config: MemoryPolicyConfig = None,
            persist_dir: str = None,
            top_k: int = 3,
            similarity_threshold: float = 0.6,
    ):
        """
        初始化长期记忆
        :param policy_config: 读写策略配置，复用全局策略
        :param persist_dir: 向量库持久化目录
        :param top_k: 单次召回返回的最大记忆条数
        :param similarity_threshold: 相似度阈值，低于该值的记忆不召回
        """
        # 1. 初始化记忆策略混入类（和 SummaryMemory 调用方式完全一致）
        self._init_policy(policy_config)

        # 2. 初始化向量存储层（底层存取完全复用VectorStore）
        if persist_dir is None:
            # 如果调用方没传，就用项目根目录下的绝对路径
            persist_dir = get_abs_path("chroma_db/long_term")
        else:
            # 如果调用方传了相对路径，也转成基于项目根目录的绝对路径
            if not os.path.isabs(persist_dir):
                persist_dir = get_abs_path(persist_dir)

        self.vector_store = VectorStore(
            persist_dir= persist_dir,
            collection_name= "long_term_memory_v2_ollama",
            distance_metric= "cosine",
        )

        # 3. 初始化向量化客户端
        self.embedding_client = EmbeddingClient()

        # 4. 召回策略配置
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        # 5. 写入缓冲，攒批写入，减少IO，提升性能
        self._pending_buffer: List[Dict] = []
        # 缓冲阈值，攒够多少触发一次批量写入
        self._flush_threshold = 5

        logger.info(f"长期记忆初始化完成，当前记忆条目数：{self.vector_store.count()}")

    # ====== 实现BaseMemory抽象接口 ======
    @property
    def history(self) -> List[Dict]:
        """
        实现基类抽象属性
        长期记忆的权威数据来源是向量库，内存不保留全量历史
        返回空列表兼容接口，持久化由VectorStore负责
        """
        return []

    def add_message(self, message: Dict) -> None:
        """
        添加单条消息到写入缓冲
        注意：不是立刻写入向量库，先缓冲，满足条件在批量落盘
        """
        # 先过策略层，敏感检测、打元数据、去重
        processed_msg = self.policy_add_message(message)
        if processed_msg is None:
            return None

        # 只有核心事实级以上的消息，才进入待写缓冲
        msg_level = processed_msg.get("metadata", {}).get("level", MemoryLevel.PROCESS)
        if msg_level <= MemoryLevel.CORE_FACT:
            self._pending_buffer.append(processed_msg)
            logger.debug(f"消息加入长期记忆写入缓冲，当前缓冲长度：{len(self._pending_buffer)}")

        # 缓冲达到阈值，触发一次批量写入
        if len(self._pending_buffer) >= self._flush_threshold:
            self.flush()
        return None

    def add_messages(self, messages: List[Dict]) -> None:
        """批量添加消息到缓冲"""
        for msg in messages:
            self.add_message(msg)

    def get_context(self) -> List[Dict]:
        """
        基类要求的接口，但长期记忆不能独立生成上下文
        长期记忆是「补充上下文」，需要配合短期记忆使用
        真正的召回逻辑是retrieve()
        """
        return []

    def restore_history(self, messages: List[Dict]) -> None:
        """
        长期记忆不依赖外部历史：
        其权威数据源是向量库，初始化时已自动加载，无需从磁盘恢复。
        """
        return

    def clear(self) -> None:
        """清空所有长期记忆"""
        self._pending_buffer = []
        self.vector_store.clear()
        logger.info("长期记忆已清空")

    # ====== 长期记忆核心能力 ======

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """把 metadata 转成 ChromaDB 支持的格式（只保留标量/list/None）"""
        import json
        clean = {}
        for k, v in metadata.items():
            if v is None or isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, (list, tuple)):
                # list 里如果有 dict，也一并转成 JSON 字符串
                clean[k] = [
                    x if isinstance(x, (str, int, float, bool)) else json.dumps(x, ensure_ascii=False)
                    for x in v
                ]
            else:
                # dict / pydantic / 其它对象 → JSON 字符串
                try:
                    clean[k] = json.dumps(v, ensure_ascii=False, default=str)
                except Exception:
                    clean[k] = str(v)
        # IntEnum → int
        if "level" in clean:
            clean["level"] = int(clean["level"])
        return clean

    def flush(self) -> None:
        """
        把缓冲里的消息批量写入向量库
        调用时机：缓冲满、对话结束、手动强制调用
        """
        if not self._pending_buffer:
            return

        try:
            # 1. 提炼记忆条目：只存有效内容，不存完整消息结构
            documents = []
            metadatas = []
            ids = []

            for msg in self._pending_buffer:
                content = msg.get("content", "").strip()
                if not content:
                    continue

                documents.append(content)
                # 元数据保留：消息级别、角色、时间戳
                metadata = msg.get("metadata", {})
                metadata["role"] = msg.get("role", "")
                metadatas.append(self._sanitize_metadata(metadata))
                # 用内容哈希做ID，天然去重，相同内容不会重复写入
                long_memory_msg_id  = hashlib.md5(content.encode("utf-8")).hexdigest()
                if long_memory_msg_id not in ids:
                    ids.append(long_memory_msg_id)

            # 2. 批量向量化
            embeddings = self.embedding_client.embedding_batch(documents)

            # 3. 批量写入向量库
            if documents:
                self.vector_store.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings,
                )
                logger.info(f"长期记忆批量写入完成，共{len(documents)}条")

        except Exception as e:
            logger.error(f"长期记忆写入失败：{str(e)}，自动降级，不影响主流程")
        finally:
            # 无论成功失败，清空缓存，避免重复写入
            self._pending_buffer = []

    def retrieve(self, query:str) -> List[Dict]:
        """
        根据用户问题，语义召回相关的长期记忆
        :param query: 用户问题文本
        :return: 相关记忆列表，按照相似度从高到低排序
        """
        try:
            # 1. 用户问题向量化
            query_vec = self.embedding_client.embedding_completion(query)

            # 2. 向量库检索
            result = self.vector_store.search(
                query_embedding=query_vec,
                top_k=self.top_k,
            )

            # 3. 过滤低相似度结果
            # 余弦距离 = 1 - 余弦相似度，距离越小越相似
            filtered = [res for res in result if res["distance"] < 1 - self.similarity_threshold]

            logger.info(f"长期记忆召回完成，命中{len(filtered)}条相关记忆")
            return filtered

        except Exception as e:
            logger.error(f"长期记忆召回失败：{str(e)}，自动降级为无长期记忆")
            return []

    def build_memory_context(self, query:str) -> Optional[Dict]:
        """
        把召回的记忆条目，拼接成一条SYSTEM消息，方便直接插入上下文
        :param query: 用户问题
        :return: 格式化后的记忆消息，没有相关记忆则返回None
        """
        memories = self.retrieve(query)
        if not memories:
            return

        # 拼接成模型容易读的格式
        memory_text = "【相关历史记忆】\n"
        for i, mem in enumerate(memories, 1):
            memory_text += f"{i}. {mem['content']}\n"

        return {
            "role": "system",
            "content": memory_text.strip()
        }