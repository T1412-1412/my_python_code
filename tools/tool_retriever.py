from typing import Optional

import numpy as np
from client.embedding_client import EmbeddingClient
from utils.config_tool import agent_conf
from utils.logger_tool import logger


# 工具检索核心模块
class ToolRetriever:
    """
    工具检索器：根据用户问题，语义召回最相关的TopK个工具
    底层原理：文本向量化 + 余弦相似度计算
    """
    def __init__(self, tool_registry: dict, embedding_client: Optional[EmbeddingClient] = None):
        self.tool_registry = tool_registry
        self.tool_name = list(tool_registry.keys())
        self.embedding_client = embedding_client or EmbeddingClient()

        # 存储每个工具的向量，key是工具名
        self.tool_embeddings = {}

        # 初始化时，就把所有工具的向量预计算好，存储在tool_embeddings中
        self._build_index()

    def _get_embeddings(self, text: str) -> np.ndarray:
        """
        调用Embedding接口，把文本转成向量
        返回numpy数组，方便后续计算
        :param text:需要转换为向量的文本
        :return: numpy数组
        """
        text_vector = self.embedding_client.embedding_completion(text)
        # 应该构建多个向量化函数，对应不同文本，例如工具向量化，用户问题向量化等等
        # print("=== embedding返回结构 ===")
        # print(type(response.data))
        # print(response.data[:5])  # 打印前5个元素看看
        # print("======================")
        return np.array(text_vector, dtype=np.float32)

    def _build_index(self):
        """
        预构建工具向量索引
        系统启动时执行一次，把所有工具的描述转成向量存起来
        这就是「索引构建」，后续查询不用再重复计算工具向量
        """
        logger.info("开始构建工具向量索引...")
        for tool_name, info in self.tool_registry.items():
            # 索引文本 = 工具名+工具描述，召回更准
            full_text = f"{tool_name} {info['search_text']}"
            self.tool_embeddings[tool_name] = self._get_embeddings(full_text)
        logger.info(f"工具向量索引构建完成，共{len(self.tool_embeddings)}个工具")

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        原理：向量夹角越小，余弦值越大，相似度越高
        返回值范围：[-1, 1]，越接近1越相似
        :param vec1: 向量1
        :param vec2: 向量2
        :return: 余弦相似度
        """
        # 公式：点积/模长乘积
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        # 避免除以零
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def retriever(self, query: str, top_k: int = agent_conf["top_k"]) -> list:
        """
        根据用户问题，检索最相关的TopK个工具
        :param query: 用户的问题文本
        :param top_k: 返回前几个工具
        :return: 工具名列表，按相似度从高到低排序
        """
        # 1. 将用户问题转换为向量
        query_vec = self._get_embeddings(query)

        # 2. 计算问题和每个工具向量的相似度
        similarity_scores = []
        for tool_name, tool_vec in self.tool_embeddings.items():
            similarity = self._cosine_similarity(query_vec, tool_vec)
            similarity_scores.append((tool_name, similarity))

        # 3. 按相似度从高到低排序
        similarity_scores.sort(key=lambda x: x[1], reverse=True)

        # 4. 取前topK个工具
        top_tools = similarity_scores[:top_k]

        # 打印调试信息，方便观察召回效果
        for name, score in top_tools:
            logger.info(f"   {name} - 相似度：{score:.4f}")
        # 只返回工具名列表
        return [name for name, score in top_tools]