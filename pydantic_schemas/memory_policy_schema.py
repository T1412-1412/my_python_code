"""
记忆读写策略相关schem
作用：结构化定义消息元数据，策略配置，避免硬编码，所有记忆类统一遵循
"""

from enum import IntEnum
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any, List


class MemoryLevel(IntEnum):
    """消息级别枚举，数字越小优先级越高，存活时间越长"""
    SYSTEM = 0      # L0 系统级：最高优先级，永不过期
    CORE_FACT = 1   # L1 核心事实级：永久归档
    PROCESS = 2     # L2 过程级：会话内有效
    TEMP = 3        # L3 临时级：单轮后丢弃
    SENSITIVE = -1  # L4 敏感级：禁止存储

class MessageMetadata(BaseModel):
    """每条消息附加的元数据，写入记忆时自动打标"""
    level: MemoryLevel = Field(default=MemoryLevel.PROCESS, description="消息级别")
    timestamp: float = Field(description="写入时间戳")
    enable_summarize: bool = Field(default=True, description="是否可以被摘要压缩")
    enable_persist: bool = Field(default=True, description="是否可以持久化到磁盘")
    task_id: Optional[str] = Field(default=None, description="关联的任务ID，用于多任务隔离")
    extra: Dict[str, Any] = Field(default_factory=dict, description="扩展字段")

class MemoryPolicyConfig(BaseModel):
    """读写策略全局配置"""
    # 敏感信息正则匹配规则（可扩展，工业级一般是多层级敏感检测）
    sensitive_patterns: list[str] = Field(
        default=[
            r"(password|passwd|pwd|密钥|api[_-]?key|token)\s*[:=]\s*\S+",
            r"\d{17}[\dXx]",    # 身份证
            r"1[3-9]\d{9}",     # 手机号
            r"\d{16,19}"        # 银行卡
        ],
        description="敏感信息正则匹配规则，匹配到的内容自动脱敏"
    )

    # ========== 新增：评级规则配置 ==========
    # 升级为核心事实的关键词（强匹配）
    core_fact_keywords: List[str] = Field(
        default=[
            "你记住", "帮我记", "别忘了", "保存下来", "记一下",
            "我叫", "我是", "我的名字", "手机号", "电话", "身份证",
            "我喜欢", "我不喜欢", "我习惯", "我的偏好",
            "必须", "一定要", "注意", "不能", "禁止",
            "是的", "没错", "对，就是", "确认"
        ],
        description="命中则直接升级为核心事实级的关键词"
    )
    # 降级为临时级的关键词（强匹配）
    temp_keywords: List[str] = Field(
        default=[
            "错误", "失败", "异常", "重试", "出错了",
            "我想想", "我分析一下", "大概", "可能", "或许",
            "等一下", "稍等", "测试一下"
        ],
        description="命中则直接降级为临时级的关键词"
    )

    # 各级别消息的上下文token配额（总配额按比例分配）
    level_token_quota: Dict[MemoryLevel, int] = Field(
        default={
            MemoryLevel.SYSTEM: 3000,
            MemoryLevel.CORE_FACT: 3000,
            MemoryLevel.PROCESS: 3000,
            MemoryLevel.TEMP: 0  # 临时级不占长期上下文配额
        },
        description="各级别消息在上下文中的最大token配额"
    )
    # 是否自动去重
    enable_dedup: bool = Field(default=True, description="是否开启写入消息去重")
    # 临时级消息是否持久化（默认否）
    persist_temp_level: bool = Field(default=False, description="是否持久化临时级消息")