"""
知识库管理器全局单例
位置中立：工具层和 API 层都可以 import，不会形成循环依赖
"""
import threading
from typing import Optional
from rag_knowledge.kb_manager import KnowledgeBaseManager
from utils.logger_tool import logger

_instance: Optional[KnowledgeBaseManager] = None
_lock = threading.Lock()


def get_global_kb_manager() -> KnowledgeBaseManager:
    """获取全局唯一的 KnowledgeBaseManager 实例（线程安全的懒加载）"""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = KnowledgeBaseManager()
    # 临时调试
    logger.debug(f"[KB单例] id={id(_instance)}")
    return _instance