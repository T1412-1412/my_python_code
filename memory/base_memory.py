"""
记忆模块抽象基类
作用：定义所有记忆类必须实现的统一接口，Agent只依赖这个抽象类，实现解耦
对应工业级：面向接口编程，符合开闭原则（新增记忆类型不用改Agent）
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .types import ChatMessage

class BaseMemory(ABC):
    """所有记忆类的父类，必须实现以下方法和属性"""

    @property
    @abstractmethod
    def history(self) -> List[Dict]:
        """全量原始历史消息（持久化用，永远保存完整数据，不能是摘要）"""
        pass

    @abstractmethod
    def add_message(self, message: Dict) -> None:
        """添加单条消息到记忆"""
        pass

    @abstractmethod
    def add_messages(self, messages: List[Dict]) -> None:
        """批量添加消息到记忆"""
        pass

    @abstractmethod
    def get_context(self) -> List[Dict]:
        """获取传给大模型的上下文（经过压缩/截断后的最终上下文）"""
        pass

    @abstractmethod
    def restore_history(self, messages: List[Dict]) -> None:
        """
        从外部数据源恢复历史。
        语义约定：
        - 内存型记忆类应精确恢复，不重跑 policy
        - 外部存储型记忆类可空操作或自行重建
        """
        pass

    def clear(self) -> None:
        """清空记忆，可选则实现"""
        pass

    # 以下是通用辅助方法（不是抽象的，子类直接继承用）
    @staticmethod
    def _normalize(message: Dict[str, Any]|ChatMessage) -> Dict[str, Any]:
        """
        统一把输入转成 dict 格式，兼容两种输入方式。
        你现有代码里 add_message 接收的是 dict：
            self.base_memory.add_message({"role": "user", "content": query})
        但 assistant 消息是 model_dump() 出来的 dict。
        这个方法确保不管传什么进来，内部处理都是统一格式。
        """
        if isinstance(message, ChatMessage):
            return message.to_api_format()
        return message