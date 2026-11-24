"""
向量存储层 —— 长期记忆的「仓库」，负责存向量、搜向量、删向量。
为什么选 ChromaDB？
--------------------
你现在是学习阶段，目标是搞懂原理，不是上线生产。
ChromaDB 的优势：
1. 零部署：pip install chromadb 就能用，不用起 Docker、不用配账号
2. 本地持久化：数据存在本地文件夹，重启不丢
3. API 简洁：add / query / delete 三个方法就够了
4. 自动做相似度计算：你不用手写余弦相似度，它内部算好
等你理解了原理，以后生产环境想换 Qdrant / Milvus / PGVector，
接口设计是一样的（add / search / delete），换底层库不影响上层逻辑。
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional
import chromadb
from chromadb.config import Settings
from utils.logger_tool import logger
from utils.path_tool import get_abs_path
from utils.config_tool import agent_conf


class VectorStore:
    """向量数据库封装，提供长期记忆的存储和检索能力。
    这一层只做「数据存取」，不做任何业务判断。
    业务逻辑（什么时候写、写什么、怎么排序）全在 LongTermMemory 里。
    这就是分层架构的好处：改业务逻辑不动存储，换存储不动业务。
    使用示例：
        store = VectorStore(persist_dir="./chroma_db")
        store.add(documents=["今天天气不错"], metadatas=[{"type": "chat"}], ids=["msg_1"])
        results = store.search(query_text="天气怎么样", top_k=3)
    """

    def __init__(
        self,
        persist_dir: str,
        collection_name: str = "long_term_memory",
        distance_metric: str = "cosine",
    ):
        """
        参数：
            persist_dir: 数据持久化目录。Chroma 会把所有向量和文本存这里，
                         重启程序后自动加载，不用重新建索引。
            collection_name: 集合名。相当于数据库里的一张表。
                            不同用途的记忆放不同 collection，互不干扰。
            distance_metric: 相似度计算方式。
                            cosine：余弦相似度，文本检索首选
                            l2：欧氏距离
                            ip：内积
        """
        if persist_dir is None:
            # 如果调用方没传，就用项目根目录下的绝对路径
            persist_dir = get_abs_path(agent_conf["chroma_db"])
        else:
            # 如果调用方传了相对路径，也转成基于项目根目录的绝对路径
            if not os.path.isabs(persist_dir):
                persist_dir = get_abs_path(persist_dir)
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.distance_metric = distance_metric

        # 确保持久化目录存在
        os.makedirs(persist_dir, exist_ok=True)

        # 初始化 Chroma 持久化客户端
        # Settings 的 anonymized_telemetry=False 是关掉匿名统计上报
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # 获取或创建 collection
        # get_or_create_collection = 没有就建一个，有就直接用
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric},
        )
        logger.info(f"向量存储初始化完成，集合: {collection_name}，当前条目数: {self.count()}")

    def count(self) -> int:
        """返回当前集合里有多少条记忆。"""
        return self.collection.count()

    def add(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """批量插入记忆条目。
        参数：
            documents: 要存储的文本内容列表
            metadatas: 每条文本对应的元数据（类型、时间、重要性等）
            ids: 每条的唯一 ID（由上层生成，比如 MemoryEntry.id）
            embeddings: 预计算好的向量。如果传了就用这个，
                       没传 Chroma 会自己调默认 embedding 模型（我们不用它默认的）
        为什么让上层传入 embeddings？
        - 我们用的是自己的 EmbeddingClient（OpenAI 模型）
        - Chroma 自带的默认 embedding 是本地小模型，效果不如 OpenAI
        - 所以我们在外面算好向量传进来，Chroma 只负责存和搜
        """
        if not documents:
            return

        # 自动补全 ids（如果没传）
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in documents]

        # 自动补全 metadatas
        if metadatas is None:
            metadatas = [{} for _ in documents]

        try:
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings,
            )
            logger.debug(f"向量存储写入 {len(documents)} 条，当前总数: {self.count()}")
        except Exception as e:
            logger.error(f"向量存储写入失败: {str(e)}")
            raise

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """根据向量相似度检索最相关的记忆。
        参数：
            query_embedding: 查询向量（由 EmbeddingClient 把用户问题转成向量）
            top_k: 返回前 K 条最相似的
            where: 元数据过滤条件，比如 {"memory_type": "long_term"}
                   可以只搜长期记忆，不搜知识库
        返回：
            列表，每个元素是一条记忆：
            {
                "id": "...",
                "content": "...",
                "metadata": {...},
                "distance": 0.234,   # 距离越小越相似
            }
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
            )

            # 把 Chroma 复杂的返回结构拍平成我们需要的格式
            hits = []
            if results and results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    hits.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i] if results["documents"] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0.0,
                    })
            logger.debug(f"向量检索完成，query 命中 {len(hits)} 条")
            return hits

        except Exception as e:
            logger.error(f"向量检索失败: {str(e)}")
            return []

    def delete(self, ids: List[str]) -> None:
        """按 ID 删除记忆。"""
        try:
            self.collection.delete(ids=ids)
            logger.debug(f"向量存储删除 {len(ids)} 条")
        except Exception as e:
            logger.error(f"向量删除失败: {str(e)}")

    def clear(self) -> None:
        """清空整个集合。"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric},
            )
            logger.info("向量存储已清空")
        except Exception as e:
            logger.error(f"清空向量存储失败: {str(e)}")