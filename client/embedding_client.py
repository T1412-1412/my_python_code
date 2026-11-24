# from __future__ import annotations
# import hashlib
# import time
# from typing import List, Optional
# from client.base_client import get_base_client
# from utils.config_tool import agent_conf
# from utils.logger_tool import logger
#
#
# class EmbeddingClient:
#     def __init__(self, enable_cache: bool = True):
#         self.client = get_base_client()
#         self.model = agent_conf["embedding_model"]
#         # 缓存：避免重复文本重复调 API（省钱）
#         self.enable_cache = enable_cache
#         # 缓存字典：key = 文本的 hash，value = 向量
#         # 为什么用 hash 当 key 而不是原文？省内存，避免超长文本做 key
#         self._cache: dict[str, List[float]] = {}
#
#     def _text_hash(self, text: str) -> str:
#         """对文本做 MD5 哈希，作为缓存 key。"""
#         return hashlib.md5(text.encode("utf-8")).hexdigest()
#
#     def embedding_completion(self, text: str, model=agent_conf["embedding_model"]):
#         """
#         把单条文本转成向量。
#         底层原理：
#         1. 先查缓存，命中就直接返回（不花 API 钱）
#         2. 没命中就调 OpenAI Embedding API
#         3. 结果写入缓存
#         4. 返回 1536 维浮点数组
#         """
#         # 如果是空文本，返回零向量，避免API报错
#         if not text or not text.strip():
#             return [0.0] * 1536
#
#         cache_key = self._text_hash(text)
#         if self.enable_cache and cache_key in self._cache:
#             logger.debug("从缓存返回向量")
#             return self._cache[cache_key]
#
#         # 调用 OpenAI Embedding API
#         start = time.time()
#         try:
#             response = self.client.embeddings.create(
#                 model=model,
#                 input=text
#             )
#             vector = response.data[0].embedding
#             spend = time.time() - start
#             logger.debug(f"文本向量化完成，耗时{spend:.2f}秒，维度{len(vector)}")
#             if self.enable_cache:
#                 self._cache[cache_key] = vector
#             return vector
#         except Exception as e:
#             logger.error(f"向量化失败：{e}")
#             # 降级：返回零向量而不是直接抛异常，让上层能继续跑
#             return [0.0] * 1536
#
#     def embedding_batch(self, texts: List[str]) -> List[List[float]]:
#         """
#         把多条文本转成向量。
#         注意：这里没有做缓存，因为批量场景通常每条只 embed 一次，
#         缓存意义不大。如果真要加，可以按单条逐个查缓存再批量补调。
#         """
#         if not texts:
#             return []
#         # 将空文本替换为零向量
#         # 记录哪些位置是空文本
#         empty_indices = {i for i, t in enumerate(texts) if not t or not t.strip()}
#         non_empty_texts = [t for i, t in enumerate(texts) if i not in empty_indices]
#
#         vectors: List[Optional[List[float]]] = [None] * len(texts)
#
#         if non_empty_texts:
#             try:
#                 response = self.client.embeddings.create(
#                     model=self.model,
#                     input=non_empty_texts,
#                 )
#                 # 建立 non_empty_texts 索引到原始位置的映射
#                 non_empty_index_map = [
#                     i for i in range(len(texts)) if i not in empty_indices
#                 ]
#                 for item in sorted(response.data, key=lambda x: x.index):
#                     original_index = non_empty_index_map[item.index]
#                     vectors[original_index] = item.embedding
#             except Exception as e:
#                 logger.error(f"批量 Embedding 失败: {str(e)}")
#
#         # 空位置填零向量
#         dim = 1536  # 或者从模型配置读取
#         zero = [0.0] * dim
#         return [v if v is not None else zero for v in vectors]

from __future__ import annotations
import hashlib
import time
from typing import List, Optional

from openai import OpenAI

from utils.config_tool import agent_conf
from utils.logger_tool import logger


class EmbeddingClient:
    """向量化客户端（本地 Ollama 版本）"""

    def __init__(self, enable_cache: bool = True):
        # ★ 关键改动 1：独立构造一个指向 Ollama 的客户端
        # 不要用 get_base_client()，那是给云端 LLM 服务商用的
        self.client = OpenAI(
            api_key="ollama",                       # Ollama 不校验，占位非空即可
            base_url=agent_conf["ollama_emb_url"],  # http://localhost:11434/v1
        )
        # ★ 关键改动 2：模型名走 ollama_emb_model
        self.model = agent_conf["ollama_emb_model"]
        # ★ 关键改动 3：维度从配置读，别硬编码 1536
        self.dim = int(agent_conf.get("embedding_dim", 1024))

        self.enable_cache = enable_cache
        self._cache: dict[str, List[float]] = {}

    def _text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _zero_vector(self) -> List[float]:
        return [0.0] * self.dim

    def _calibrate_dim(self, vec: List[float]) -> None:
        """首次拿到真实向量时校准维度，防止配置写错导致零向量维度对不上"""
        if len(vec) != self.dim:
            logger.warning(
                f"embedding 实际维度 {len(vec)} 与配置 {self.dim} 不一致，"
                f"已自动校准为 {len(vec)}"
            )
            self.dim = len(vec)

    def embedding_completion(self, text: str, model: str = None):
        if not text or not text.strip():
            return self._zero_vector()

        cache_key = self._text_hash(text)
        if self.enable_cache and cache_key in self._cache:
            logger.debug("从缓存返回向量")
            return self._cache[cache_key]

        model = model or self.model
        start = time.time()
        try:
            response = self.client.embeddings.create(model=model, input=text)
            vector = response.data[0].embedding
            self._calibrate_dim(vector)
            cost = time.time() - start
            logger.debug(f"本地向量化完成，耗时{cost:.2f}s，维度{len(vector)}")
            if self.enable_cache:
                self._cache[cache_key] = vector
            return vector
        except Exception as e:
            logger.error(f"本地向量化失败：{e}，返回零向量降级")
            return self._zero_vector()

    def embedding_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        empty_indices = {i for i, t in enumerate(texts) if not t or not t.strip()}
        non_empty = [t for i, t in enumerate(texts) if i not in empty_indices]
        vectors: List[Optional[List[float]]] = [None] * len(texts)

        if non_empty:
            try:
                response = self.client.embeddings.create(model=self.model, input=non_empty)
                non_empty_idx_map = [i for i in range(len(texts)) if i not in empty_indices]
                for item in sorted(response.data, key=lambda x: x.index):
                    original_idx = non_empty_idx_map[item.index]
                    vectors[original_idx] = item.embedding
                    self._calibrate_dim(item.embedding)
            except Exception as e:
                # ★ 降级：不同 Ollama 版本对 batch 的支持有差异，批量失败就逐条补
                logger.warning(f"批量向量化失败：{e}，降级为逐条调用")
                for i, t in zip(
                    [j for j in range(len(texts)) if j not in empty_indices],
                    non_empty,
                ):
                    vectors[i] = self.embedding_completion(t)

        zero = self._zero_vector()
        return [v if v is not None else zero for v in vectors]