"""
记忆模块相关的Pydantic Schema
作用：统一校验记忆配置、摘要结果，避免硬编码，参数错误提前暴露
对应工业级：所有可变配置必须结构化校验，不能用裸字典
"""

from pydantic import BaseModel, Field


class MemoryConfig(BaseModel):
    """记忆模块统一配置，所有记忆类都用这个结构传参"""
    # 保留最近多少轮原文（混合模式核心参数，太大会费token，太小会丢失细节）
    keep_recent_rounds: int = Field(default=6, ge=2, le=20, description="保留最近N轮完整原文")
    # 触发摘要的阈值：当前上下文总token超过这个值就触发压缩
    summary_token_threshold: int = Field(default=9000, ge=5000, description="触发摘要的token阈值")
    # 摘要触发的轮次：每累计N轮旧对话就触发一次压缩（和阈值双触发）
    summary_round_interval: int = Field(default=4, ge=2, description="每N轮旧对话触发一次摘要")
    # 摘要最大长度（限制模型输出的摘要不要太长）
    max_summary_tokens: int = Field(default=1500, ge=100, le=2500, description="摘要最大token数")
    # 是否开启增量摘要（工业级默认开启，关闭则每次全量摘要，耗费较多token）
    enable_incremental_summary: bool = Field(default=True, description="是否开启增量摘要")

class SummaryResult(BaseModel):
    """对得到的摘要结果进行结构化，方便缓存和调试"""
    summary_text: str = Field(description="摘要的文本内容")
    covered_message_count: int = Field(description="本次摘要覆盖了多少条原始消息")
    total_tokens: int = Field(description="摘要本身的token数")
    created_at: float = Field(description="摘要生成的时间戳")