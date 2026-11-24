"""
记忆系统的【数据结构层】——所有模块共用的基础类型定义
在工业级代码里，“数据长什么样”必须先于“怎么处理数据”被定义清楚。
否则 A 模块用dict，B 模块用dataclass，C 模块用pydantic model，
最后互相传参时全是隐式转换，有很多bug隐患
这里只做一件事：把记忆系统里的"名词"定义清楚，后面所有文件都引用它
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# 1. 消息角色枚举 —— 替代到处乱飞的字符串 "user" / "assistant" / "tool"
class MessageRole(str, Enum):
    """
    对话消息的角色类型。
    为什么用 Enum 而不是裸字符串？
    - 拼错了（比如 "assitant"）IDE 不会报错，运行时才炸；
    - 用 Enum 后，打错字直接语法检查过不去，编译期就发现问题。
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

# 2. 记忆类型枚举 —— 区分短期/长期/摘要等不同记忆
class MemoryType(str, Enum):
    """记忆的分类，用于向量库 metadata 过滤和统计。"""
    SHORT_TERM = "short_term"   # 短期窗口里的原始对话
    LONG_TERM = "long_term"     # 从短期溢出后压缩写入的长期记忆
    SUMMARY = "summary"         # 摘要记忆
    KNOWLEDGE = "knowledge"     # 外部知识库召回的事实

# 3. 单条记忆条目 —— 长期记忆的最小存储单元
@dataclass
class MemoryEntry:
    """
    一条长期记忆的完整结构。
    为什么不用简单的 dict 而要建 dataclass？
    - dict 的 key 是字符串，打错了运行时才报错；
    - dataclass 的字段是强类型，IDE 自动补全 + 类型检查；
    - 序列化/反序列化有明确方法，不用手写 json.dumps。
    字段设计思考：
    - id: 唯一标识，用于删除/更新
    - content: 记忆正文（要向量化的文本）
    - memory_type: 这条记忆属于哪一类
    - metadata: 附加信息（来源对话轮次、重要性分数、标签等）
    - created_at: 写入时间，用于过期淘汰和时间衰减
    - importance: 重要性分数（0~1），越重要越不容易被淘汰
    """
    content: str
    memory_type: MemoryType = MemoryType.LONG_TERM
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5     # 默认中等重要性
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """将 MemoryEntry 类对象序列化为 dict，方便存入 Chroma / JSON 文件。"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "metadata": self.metadata,
            "importance": self.importance,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """从 dict 反序列化为 MemoryEntry 类对象。"""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            metadata=data.get("metadata", {}),
            importance=data.get("importance", 0.5),
            created_at=data.get("created_at", time.time()),
        )

# 4. 对话消息 —— 短期记忆里的最小单元（兼容 OpenAI 格式）
@dataclass
class ChatMessage:
    """
    一条标准对话消息，格式对齐 OpenAI Chat Completion API。
    为什么要对齐 OpenAI 格式？
    - 你的 LLMClient 直接把 messages 数组传给 OpenAI 兼容接口；
    - 如果内部格式和 API 格式不一致，每次发送前都要做一次转换，
      多一层转换就多一个 bug 源。直接对齐，零转换成本。
    """
    role: MessageRole
    content: str
    # tool 类型消息专用字段
    tool_call_id: Optional[str] = None
    # assistant 消息可能携带的工具调用（结构化）
    tool_calls: Optional[List[Dict[str, Any]]] = None
    # 附加元数据（不发给模型，只本地用）
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_api_format(self) -> Dict[str, Any]:
        """将 ChatMessage 类对象转换为发给 LLM API 的 dict 格式。"""
        msg: Dict[str, Any] = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg

    @classmethod
    def from_api_format(cls, data: Dict[str,Any]) -> "ChatMessage":
        """从 API 返回的 dict 构造（比如 model_dump() 出来的 assistant 消息）转换为 ChatMessage 类对象。"""
        role = MessageRole(data["role"])
        return cls(
            role= role,
            content= data.get("content", ""),
            tool_call_id= data.get("tool_call_id"),
            tool_calls= data.get("tool_calls"),
        )