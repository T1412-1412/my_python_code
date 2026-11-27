"""
知识库管理器：对外提供完整的知识库操作能力
整合文档加载、文本分片、向量存储，是业务层和存储层的中间层
完全复用你现有的 VectorStore 和 EmbeddingClient
"""
import os
from typing import List, Dict
from rag_knowledge.document_loader import DocumentLoader
from rag_knowledge.text_splitter import RecursiveCharacterTextSplitter
from rag_knowledge.vector_store import VectorStore
from client.embedding_client import EmbeddingClient
from utils.logger_tool import logger
from utils.path_tool import get_abs_path
import re
from typing import Optional

# 知识库名称规则：小写字母开头，3-63 位，允许小写字母/数字/下划线
# 强制比 ChromaDB 的约束更严，保证"用户能建的名字，一定能被 ChromaDB 接受"
_KB_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,62}$")


def _is_valid_kb_name(name: str) -> bool:
    return bool(_KB_NAME_PATTERN.match(name))


class KnowledgeBaseManager:
    """
    知识库管理器：一个知识库对应向量库中的一个collection
    支持：创建知识库、上传文档、检索内容、删除知识库
    """

    def __init__(
            self,
            base_persist_dir: str = "chroma_db/knowledge_bases",
            chunk_size: int = 500,
            chunk_overlap: int = 100,
    ):
        """
        初始化知识库管理器
        :param base_persist_dir: 所有知识库的根目录，每个知识库一个子目录
        """
        self.base_persist_dir = get_abs_path(base_persist_dir)
        os.makedirs(self.base_persist_dir, exist_ok=True)

        # 复用全局向量化客户端
        self.embedding_client = EmbeddingClient()

        # 初始化分片器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # 缓存已经加载的向量库实例，避免重复初始化
        self._store_cache: Dict[str, VectorStore] = {}
        logger.info("知识库管理器初始化完成")

    def _get_store(self, kb_name: str, create_if_not_exists: bool = True) -> Optional[VectorStore]:
        """
        获取指定知识库的向量存储实例，懒加载+缓存
        :param create_if_not_exists: 不存在时是否自动创建，检索时传False，上传时传True
        """
        # ★ 补强 1：格式校验放在最前面，非法名不触发任何文件系统操作
        #    中文残留目录（设备采购/采购方案）在入口就被拦截，不会走到 VectorStore
        if not _is_valid_kb_name(kb_name):
            logger.warning(f"知识库名称不合法，拒绝访问：{kb_name!r}")
            return None
        if kb_name not in self._store_cache:
            persist_dir = os.path.join(self.base_persist_dir, kb_name)

            # ★ 补强 2：用 isdir 代替 exists，防止同名文件造成误判
            # 检索场景：目录不存在直接返回 None，不自动创建
            if not create_if_not_exists and not os.path.isdir(persist_dir):
                return None

            self._store_cache[kb_name] = VectorStore(
                persist_dir=persist_dir,
                collection_name=f"kb_{kb_name}",
                distance_metric="cosine",
            )
        return self._store_cache[kb_name]

    def add_document(self, kb_name: str, file_path: str) -> int:
        """
        向指定知识库添加一个文档
        :param kb_name: 知识库名称
        :param file_path: 本地文件路径
        :return: 成功添加的分片数量，失败返回0
        """
        # ★ 补强 3：上传前先做格式校验 + 存在性校验
        #    知识库的创建应该走显式的 create_kb()（或 API 创建接口），
        #    而不是"上传时顺带创建"
        if not _is_valid_kb_name(kb_name):
            raise ValueError(f"知识库名称不合法：{kb_name!r}")
        if not os.path.isdir(os.path.join(self.base_persist_dir, kb_name)):
            raise FileNotFoundError(
                f"知识库 '{kb_name}' 不存在，请先通过创建接口创建"
            )

        store = self._get_store(kb_name, create_if_not_exists=False)
        if store is None:
            raise FileNotFoundError(f"知识库 '{kb_name}' 无法打开")
        # 1. 加载文档提取纯文本
        text = DocumentLoader.load_file(file_path)
        if not text:
            return 0

        # 2. 文本分片
        chunks = self.text_splitter.split_text(text)
        if not chunks:
            return 0

        # 3. 批量向量化
        embeddings = self.embedding_client.embedding_batch(chunks)

        # 4. 生成ID：用内容哈希，天然去重
        import hashlib
        ids = [hashlib.md5(chunk.encode("utf-8")).hexdigest() for chunk in chunks]

        # 5. 元数据：记录来源文件名
        metadatas = [
            {"source": os.path.basename(file_path), "chunk_index": i}
            for i, chunk in enumerate(chunks)
        ]

        # 6. 写入向量库
        store = self._get_store(kb_name)
        store.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

        logger.info(f"知识库[{kb_name}] 添加文档成功：{os.path.basename(file_path)}，共{len(chunks)}个分片")
        return len(chunks)

    def search(self, kb_name: str, query: str, top_k: int = 3) -> List[Dict]:
        """
        在指定知识库中检索相关内容
        :param kb_name: 知识库名称
        :param query: 查询问题
        :param top_k: 返回最相关的前K条
        :return: 相关片段列表，包含内容、来源、相似度
        """
        store = self._get_store(kb_name)
        # ★ 用 is None 显式判断，虽然 VectorStore 没有 __bool__，
        #   但显式写法更防御，未来加 __len__ 也不会改变语义
        if store is None:
            logger.warning(f"知识库 [{kb_name}] 不存在，无法检索")
            return []

        # 1. 查询向量化
        query_vec = self.embedding_client.embedding_completion(query)

        # 2. 向量检索
        results = store.search(query_embedding=query_vec, top_k=top_k)

        # 3. 格式化返回
        formatted = []
        for res in results:
            formatted.append({
                "content": res["content"],
                "source": res.get("metadata", {}).get("source", "unknown"),
                "score": round(1 - res["distance"], 4),  # 距离转相似度，0~1，越大越相似
            })

        logger.debug(f"知识库[{kb_name}]检索完成，命中{len(formatted)}条相关内容")
        return formatted

    def delete_kb(self, kb_name: str) -> bool:
        """删除整个知识库"""
        if kb_name in self._store_cache:
            self._store_cache.pop(kb_name)
        # 注意：VectorStore的clear只清空集合，不删除目录
        try:
            store = self._get_store(kb_name)
            store.clear()
            logger.info(f"知识库[{kb_name}] 已清空")
            return True
        except Exception as e:
            logger.error(f"删除知识库失败：{e}")
            return False

    def list_kbs(self) -> List[str]:
        """获取所有知识库名称列表"""
        if not os.path.exists(self.base_persist_dir):
            return []
        # 列出所有子目录，每个子目录对应一个知识库
        return [
            name for name in os.listdir(self.base_persist_dir)
            if os.path.isdir(os.path.join(self.base_persist_dir, name))
        ]
