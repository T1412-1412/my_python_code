"""
摘要记忆核心实现（混合模式：历史摘要 + 最近K轮原文）
核心特性：全量历史永久保留，增量摘要，双触发策略，失败自动降级
"""

import json
import os
import time
from typing import List, Dict, Optional
from memory.base_memory import BaseMemory
from memory.summary_generator import Summarizer
from pydantic_schemas.memory_schemas import MemoryConfig, SummaryResult
from utils.config_tool import agent_conf
from utils.count_tokens import count_tokens
from utils.logger_tool import logger
from memory.memory_policy import MemoryPolicyMixin
from pydantic_schemas.memory_policy_schema import MemoryPolicyConfig

class SummaryMemory(BaseMemory, MemoryPolicyMixin): # 多继承，混入类
    def __init__(self, llm_client , chat_model: str, config: MemoryConfig, policy_config: MemoryPolicyConfig=None):
        # 初始化策略，在记忆类的__init__中调用一次即可
        self._init_policy(policy_config)
        self.config = config
        # 全量原始历史，永远完整存储对话历史，持久化也存这个，绝不丢失原始数据
        self._history: List[Dict] = []
        # 摘要缓存，当前已经生成的历史摘要，只用来构建上下文，不参与持久化
        self._current_summary: Optional[SummaryResult] = None
        # 已经被摘要覆盖的消息条数，用来判断哪些新消息需要被压缩
        self._summarized_count: int = 0
        # 摘要生成器，独立封装
        self.summarizer = Summarizer(
            llm_client=llm_client,
            chat_model=chat_model,
            max_summary_tokens=config.max_summary_tokens
        )

    @property
    def history(self) -> List[Dict]:
        """实现基类抽象属性，返回全量原始历史，给agent持久化使用"""
        return self._history.copy() # 返回副本，防止外部修改内部状态

    def add_message(self, message: Dict) -> None:
        """添加单条消息，添加后检查是否需要触发摘要"""
        # 消息写入前策略处理：敏感信息检测 -> 添加元数据 -> 去重，返回处理后的消息
        policy_message = self.policy_add_message(message)
        if policy_message is not None:
            self._history.append(policy_message)
        # 每次添加消息后，检查是否需要压缩


    def add_messages(self, messages: List[Dict]) -> None:
        """批量添加消息，批量添加后只检查一次摘要，避免重复触发"""
        # 消息写入前策略处理：敏感信息检测 -> 添加元数据 -> 去重，返回处理后的消息
        policy_messages = []
        for msg in messages:
            policy_msg = self.policy_add_message(msg)
            if policy_msg is not None:  # ✅ 判空，过滤掉 None
                policy_messages.append(policy_msg)

        if policy_messages:
            self._history.extend(policy_messages)
        self._maybe_trigger_summary()

    def _get_unsummarized_messages(self) -> List[Dict]:
        """
        获取需要被压缩的旧消息：除了最近K轮原文外、且还没有被摘要的消息
        核心逻辑：最近K轮永远保留原文，更早的没有摘要的消息就拿去压缩
        """
        # 最近K轮的起始索引：总消息数 - 保留轮数*2（因为一轮对话是user + assistant/tool两条消息）
        # max的作用是，如果总消息数很少，那么最近K轮的起始索引就可能为负数，这里用max确保不会为负数
        recent_strat_idx = max(0, len(self._history) - self.config.keep_recent_rounds*2)
        # 需要压缩的消息：从已摘要结束的位置 到 最近K轮开始的位置
        unsummarized = self._history[self._summarized_count:recent_strat_idx]

        return unsummarized

    def _maybe_trigger_summary(self) -> None:
        """摘要双触发检查：满足轮次条件或者token阈值就触发摘要"""
        unsummarized = self._get_unsummarized_messages()
        if not unsummarized:
            return None # 如果没有要压缩的消息，直接返回

        # 触发条件1：未摘要的消息达到了设定的轮次间隔
        round_trigger = len(unsummarized) >= self.config.summary_round_interval*2
        # 触发条件2：未摘要消息的 token 超过阈值（改这里！）/当前总上下文token超过阈值
        unsummarized_tokens = count_tokens(unsummarized)
        token_trigger = unsummarized_tokens >= self.config.summary_token_threshold

        if round_trigger or token_trigger:
            logger.info(f"触发摘要：未摘要消息{len(unsummarized)}条，未摘要token{unsummarized_tokens}")
            self._do_summary(unsummarized)
        return None

    # 新增显式入口，由 Agent 在事务边界调用
    def maybe_summary(self) -> None:
        """由 Agent 在每轮对话结束/任务结束时显式调用"""
        self._maybe_trigger_summary()

    def _do_summary(self, unsummarized: List[Dict]) -> None:
        """执行摘要，增量更新摘要缓存"""
        existing_summary_text = self._current_summary.summary_text if self._current_summary else ""
        is_full_summary = not self.config.enable_incremental_summary
        if is_full_summary:
            # 关闭增量摘要：从头覆盖所有旧消息
            existing_summary_text = ""
            unsummarized = self._history[: max(0, len(self._history) - self.config.keep_recent_rounds * 2)]

        new_summary = self.summarizer.generate_summary(existing_summary_text, unsummarized)
        if new_summary:
            self._current_summary = new_summary
            if is_full_summary:
                self._summarized_count = len(unsummarized)  # 全量模式：覆盖
            else:
                self._summarized_count += len(unsummarized)  # 增量模式：累加
            # 统计token节省比例
            # 整体上下文情况（辅助观察）
            full_history_tokens = count_tokens(self._history)
            current_context_tokens = count_tokens(self.get_context())
            save_ratio = current_context_tokens / full_history_tokens if full_history_tokens > 0 else 0
            # ===== 摘要压缩效率指标 =====
            covered_messages = self._history[: self._summarized_count]
            covered_tokens = count_tokens(covered_messages)  # 被摘要覆盖的原始 token
            summary_tokens = new_summary.total_tokens  # 摘要本身 token（生成时已算好）
            compression = summary_tokens / covered_tokens if covered_tokens > 0 else 0
            logger.info(
                f"摘要更新完成 | 覆盖 {self._summarized_count} 条原始消息({covered_tokens} token) "
                f"→ 摘要 {summary_tokens} token，压缩率 {compression:.1%} | "
                f"当前上下文 {current_context_tokens} token，占全量历史 {save_ratio:.1%}"
            )
        # 摘要失败的话，什么都不做，自动降级为全量保留最近K轮（滑动窗口模式）
    def get_context(self) -> List[Dict]:
        """
        构建传给大模型的最终上下文：固定结构 [system, 摘要, 最近K轮原文]
        注意：system消息永远不进摘要，永远放在第一条
        """
        context = []
        # 1. 先放system消息（永远第一条，不参与摘要）
        if self._history and self._history[0]["role"] == "system":
            context.append(self._history[0])
        else:
            context.append({"role": "system", "content": agent_conf["system_prompt"]})

        # 2. 添加历史摘要（如果有的话就加入，没有就忽略）
        if self._current_summary:
            context.append({
                "role": "system",   # 用system角色放摘要，模型会更重视
                "content": f"【历史对话摘要】\n{self._current_summary.summary_text}"
            })

        # 3. 添加最近K轮原文（包含工具调用和结果，保证细节）
        recent_strat_idx = max(0, len(self._history) - self.config.keep_recent_rounds * 2)
        # 如果第一条是system，要跳过，避免重复放system
        if self._history and self._history[0]["role"] == "system":
            recent_strat_idx = max(1, recent_strat_idx)
        context.extend(self._history[recent_strat_idx:])
        # 过滤上下文窗口：读取时策略处理：1.按优先级排序、2.分级配额淘汰、3.剥离元数据
        policy_context = self.policy_filter_context(context)
        return policy_context

    def restore_history(self, messages: List[Dict]) -> None:
        """精确恢复：不重跑 policy、不触发摘要，直接灌入 _history"""
        if not messages:
            return
        cleaned = [m for m in messages if isinstance(m, dict)]
        self._history = cleaned # 追加改为赋值，安全，也是为了状态机的恢复
        logger.info(f"恢复历史 {len(cleaned)} 条到 SummaryMemory")

    def save_state(self, path: str) -> bool:
        """
        持久化摘要恢复所需的完整状态。
        存 policy 后的 _history（与 _summarized_count 索引对齐）+ 摘要 + 覆盖计数
        """
        try:
            data = {
                "history": self._history,  # ← 注意是 _history，不是外部传入的 session_log
                "summarized_count": self._summarized_count,
                "summary": self._current_summary.model_dump() if self._current_summary else None,
                "saved_at": time.time(),
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            logger.info(
                f"摘要状态已保存：历史 {len(self._history)} 条，"
                f"摘要覆盖 {self._summarized_count} 条"
            )
            return True
        except Exception as e:
            logger.error(f"保存摘要状态失败：{e}")
            return False

    def load_state(self, path: str) -> bool:
        """
        恢复摘要状态。返回 True 时 Agent 不应再调用 restore_history。
        """
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            history = [m for m in data.get("history", []) if isinstance(m, dict)]
            if not history:
                return False

            # 一致性校验：summarized_count 必须在 [0, len(history)] 内
            summarized_count = data.get("summarized_count", 0)
            summarized_count = max(0, min(summarized_count, len(history)))

            self._history = history  # 注意是覆盖（=），不是 extend
            self._summarized_count = summarized_count
            if data.get("summary"):
                self._current_summary = SummaryResult(**data["summary"])

            logger.info(
                f"摘要状态恢复：历史 {len(self._history)} 条，"
                f"摘要覆盖 {self._summarized_count} 条，"
                f"摘要存在={self._current_summary is not None}"
            )
            return True
        except Exception as e:
            logger.error(f"恢复摘要状态失败：{e}，将退回全量重摘")
            return False

    def clear(self) -> None:
        """清空记忆，重置所有状态"""
        self._history = []
        self._current_summary = None
        self._summarized_count = 0
        logger.info("记忆已清空，所有状态重置")
