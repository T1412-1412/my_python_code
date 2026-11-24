"""
文档加载器：统一解析不同格式的文档，输出纯文本
支持格式：.txt, .md, .pdf
设计原则：输入是文件路径，输出是纯文本字符串，格式差异在内部抹平
"""
import os
from typing import Optional
from utils.logger_tool import logger


class DocumentLoader:
    """统一文档加载器，根据文件后缀自动选择解析方式"""

    @staticmethod
    def load_file(file_path: str) -> Optional[str]:
        """
        加载单个文档文件，返回纯文本内容
        :param file_path: 本地文件的绝对路径
        :return: 解析后的纯文本，失败返回None
        """
        if not os.path.exists(file_path):
            logger.error(f"文件不存在：{file_path}")
            return None

        # 获取文件后缀，小写统一判断
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext in (".txt", ".md", ".markdown"):
                return DocumentLoader._load_text(file_path)
            elif ext == ".pdf":
                return DocumentLoader._load_pdf(file_path)
            else:
                logger.warning(f"暂不支持的文件格式：{ext}")
                return None
        except Exception as e:
            logger.error(f"解析文件失败 {file_path}：{e}")
            return None

    @staticmethod
    def _load_text(file_path: str) -> str:
        """加载纯文本/Markdown文件，Python原生支持"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.debug(f"加载文本文件成功：{file_path}，长度 {len(content)} 字符")
        return content

    @staticmethod
    def _load_pdf(file_path: str) -> str:
        """加载PDF文件，提取所有页面的文本"""
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        text_list = []
        # 逐页提取文本
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_list.append(page_text.strip())

        full_text = "\n\n".join(text_list)
        logger.debug(f"加载PDF成功：{file_path}，共{len(reader.pages)}页，{len(full_text)}字符")
        return full_text
