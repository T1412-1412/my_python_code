"""
文本分片器：递归字符分片法，工业界基线方案
核心思想：优先按语义分隔符（段落、换行、句号）切分，实在太长再硬切
保留重叠字符，避免语义在切块边界处断裂
"""
from typing import List
from utils.logger_tool import logger


class RecursiveCharacterTextSplitter:
    """
    递归字符分片器：从大到小尝试分隔符，尽量保持语义完整
    对应 LangChain 的 RecursiveCharacterTextSplitter，实现思路完全一致
    """

    def __init__(
            self,
            chunk_size: int = 500,
            chunk_overlap: int = 100,
            separators: List[str] = None,
    ):
        """
        初始化分片器
        :param chunk_size: 每个分片的最大字符数，默认500字符（约300-400token）
        :param chunk_overlap: 相邻分片的重叠字符数，默认100，避免语义截断
        :param separators: 分隔符优先级列表，从左到右优先级降低
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 默认分隔符：从段落 → 行 → 句子 → 单词 → 字符，逐级降级
        if separators is None:
            self.separators = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]
        else:
            self.separators = separators

    def split_text(self, text: str) -> List[str]:
        """
        对文本进行分片，返回分片列表
        :param text: 原始长文本
        :return: 分片后的字符串列表
        """
        if not text or not text.strip():
            return []

        chunks = self._split_text(text, self.separators)
        logger.debug(f"文本分片完成：原文{len(text)}字符 → {len(chunks)}个分片")
        return chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        递归分片核心逻辑
        1. 拿当前优先级最高的分隔符切分
        2. 切完后每个小块如果小于chunk_size，就收集起来
        3. 如果还是太大，就用下一级分隔符继续切
        """
        chunks = []
        # 当前用的分隔符
        separator = separators[0]
        # 剩余的分隔符（下一级）
        remaining_separators = separators[1:] if len(separators) > 1 else [""]

        # 用当前分隔符把文本切开
        splits = text.split(separator) if separator else list(text)

        # 临时缓存：攒小的片段，凑够一个chunk就合并输出
        current_batch = []
        current_length = 0

        for split in splits:
            split_len = len(split) + len(separator)  # 加上分隔符本身的长度

            # 如果加上当前片段还没超上限，就加入缓存
            if current_length + split_len <= self.chunk_size:
                current_batch.append(split)
                current_length += split_len
            else:
                # 缓存里攒够了，输出一个chunk
                if current_batch:
                    chunk = separator.join(current_batch).strip()
                    if chunk:
                        chunks.append(chunk)

                # 处理重叠：把当前chunk末尾的一部分，挪到下一个chunk的开头
                if self.chunk_overlap > 0 and chunks:
                    # 取上一个chunk的末尾重叠部分
                    last_chunk = chunks[-1]
                    overlap_text = last_chunk[-self.chunk_overlap:]
                    current_batch = [overlap_text, split]
                    current_length = len(overlap_text) + split_len
                else:
                    # 没有重叠的话，当前片段作为新batch的起点
                    # 如果单个片段就比chunk_size还大，就递归用更细的分隔符切
                    if split_len > self.chunk_size:
                        if remaining_separators:
                            sub_chunks = self._split_text(split, remaining_separators)
                            chunks.extend(sub_chunks)
                        else:
                            # 已经到最细粒度（单个字符），直接硬切
                            chunks.append(split[:self.chunk_size])
                        current_batch = []
                        current_length = 0
                    else:
                        current_batch = [split]
                        current_length = split_len

        # 处理最后剩下的缓存
        if current_batch:
            chunk = separator.join(current_batch).strip()
            if chunk:
                chunks.append(chunk)

        return chunks
