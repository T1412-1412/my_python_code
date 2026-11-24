"""
agent主循环：规划 + FSM架构 + 中断持久化恢复
设计原则：单态单责、纯函数跳转、终态统一收口
新增能力：检查点自动保存、原子写入、断点续跑、幂等恢复
"""

from enum import Enum
from typing import List, Dict, Optional
import os
import json
import time
import tempfile

from plan_and_execute.planner import Planner
from plan_and_execute.executor import Executor
from plan_and_execute.reflector import Reflector
from plan_and_execute.plan_model import (
    Plan, PlanStatus, StepStatus,
    plan_to_dict, plan_from_dict
)
from client.llm_client import LLMClient
from tools.tool_retriever import ToolRetriever
from pydantic_schemas.build_tool_schema import build_tool_schema
from tools.run_tools_parallel import run_tools_parallel
from memory.base_memory import BaseMemory
from memory.long_term_memory import LongTermMemory
from utils.config_tool import agent_conf
from utils.logger_tool import logger
from utils.path_tool import get_abs_path


# ==============================================
# 第一部分：状态枚举 + 迁移表
# ==============================================
class AgentState(str, Enum):
    """
    Agent主循环状态枚举：所有可能的状态全部显式定义
    规则：同一时间只能有一个状态；终态只有两个：COMPLETED / FAILED
    """
    INIT = "init"                           # 初始态：接收query，做前置准备
    PLANNING = "planning"                   # 规划态：生成计划
    EXECUTING = "executing"                 # 执行态：执行计划
    RESULT_PROCESSING = "result_processing" # 结果态：校验进度，决定下一状态
    REFLECTING = "reflecting"               # 反思态：失败后重规划
    COMPLETED = "completed"                 # 终态：任务成功完成
    FAILED = "failed"                       # 终态：任务失败，走兜底

_AGENT_TRANSITIONS = {
    AgentState.INIT: {AgentState.PLANNING},
    AgentState.PLANNING: {AgentState.EXECUTING, AgentState.FAILED},
    AgentState.EXECUTING: {AgentState.RESULT_PROCESSING},
    AgentState.RESULT_PROCESSING: {
        AgentState.COMPLETED,
        AgentState.FAILED,
        AgentState.EXECUTING,
        AgentState.REFLECTING
    },
    AgentState.REFLECTING: {AgentState.EXECUTING, AgentState.FAILED},
    AgentState.COMPLETED: set(),
    AgentState.FAILED: set(),
}

# 快照版本号：后续结构变更时升级，或者兼容旧快照
SNAPSHOT_VERSION = 1

# ==============================================
# 第二部分：纯函数
# ==============================================
def _summarize_result(llm: LLMClient, plan: Plan) -> str:
    """把多步骤结果汇总成通顺回答，失败则拼接兜底"""
    done_steps = [
        f"{s.description}: {s.result}"
        for s in plan.steps
        if s.status == StepStatus.DONE
    ]
    try:
        response = llm.chat_completion(
            model=agent_conf["chat_model"],
            messages=[
                {"role": "system", "content": "把步骤结果整合成通顺完整的回答，不要罗列步骤"},
                {"role": "user", "content": f"需求：{plan.goal}\n" + "\n".join(done_steps)},
            ],
            tool_schema=[],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.warning(f"汇总失败，降级拼接：{e}")
        return "\n".join(done_steps)

# ==============================================
# 第三部分：Agent 主体
# ==============================================
class PlanAgent:
    def __init__(
            self,
            llm_client: LLMClient,
            retriever: ToolRetriever,
            planner: Planner,
            executor: Executor,
            reflector: Reflector,
            base_memory: BaseMemory,
            long_term_memory: LongTermMemory,
            snapshot_path: str,
            enable_checkpoint: bool = True,
            storage_dir: Optional[str] = None,  # ★ 新增：会话专属存储目录
    ):
        # 底层能力组件
        self.llm_client = llm_client
        self.tool_retriever = retriever
        self.planner = planner
        self.executor = executor
        self.reflector = reflector

        # 记忆体系
        self.base_memory = base_memory
        self.long_term_memory = long_term_memory
        self._session_log: List[Dict] = []
        self._history_inited = False

        # FSM 运行时状态
        self._agent_state: AgentState = AgentState.INIT
        self._query: str = ""
        self._plan: Optional[Plan] = None
        self._final_answer: str = ""
        self._ltm_context: Optional[Dict] = None

        # ========== 新增：持久化配置 ==========
        self.enable_checkpoint = enable_checkpoint
        # 默认快照路径：按会话固定存储，可外部指定
        self.snapshot_path = snapshot_path or get_abs_path("history_messages/agent_snapshot.json")
        # 会话专属的历史文件目录，默认退回全局路径（兼容单会话场景）
        self._storage_dir = storage_dir or get_abs_path("history_messages")

        # 历史加载
        self._init_history_from_disk()

    # ===================== 记忆体系 =====================
    def _load_history(self) -> List[Dict]:
        file_path = os.path.join(self._storage_dir, "history_messages.json")   # ★ 变了
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    messages = json.load(f)
                logger.debug("历史记录加载成功")
                return messages
            except Exception as e:
                logger.error(f"加载历史记录失败：{e}")
                return []
        logger.debug("历史记录不存在")
        return []

    def _persist_history(self) -> None:
        """会话历史 + 记忆快照落盘"""
        file_path = os.path.join(self._storage_dir, "history_messages.json")   # ★ 变了
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self._session_log, f, ensure_ascii=False)
            logger.info("会话记录持久化完成")
        except Exception as e:
            logger.error(f"会话记录保存失败：{e}")
        if hasattr(self.base_memory, "save_state"):
            state_path = os.path.join(self._storage_dir, "memory_state.json")  # ★ 变了
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            self.base_memory.save_state(state_path)

    def _init_history_from_disk(self) -> None:
        """从磁盘加载历史，幂等执行一次"""
        if self._history_inited:
            return
        state_path = os.path.join(self._storage_dir, "memory_state.json")  # ★ 变了
        restored = False
        if hasattr(self.base_memory, "load_state"):
            restored = self.base_memory.load_state(state_path)
        if not restored:
            history = self._load_history()
            if history:
                self._session_log = history
                self.base_memory.restore_history(history)
                logger.info(f"历史记忆注入完成，共 {len(history)} 条")
        if not self._session_log and restored:
            self._session_log = self.base_memory.history
        self._history_inited = True

    def _record_message(self, msg: Dict) -> None:
        """四路全写：上下文 + 会话日志 + 完整日志 + 长期记忆"""
        self.base_memory.add_message(msg)
        self._session_log.append(msg)
        self._append_transcript(msg)
        self.long_term_memory.add_message(msg)

    def _record_runtime(self, msg: Dict) -> None:
        """运行时中间消息：写上下文+完整日志，不污染会话和长期记忆"""
        self.base_memory.add_message(msg)
        self._append_transcript(msg)

    def _append_transcript(self, msg: Dict) -> None:
        """完整日志 JSONL 追加"""
        try:
            transcript_path = os.path.join(self._storage_dir, "transcript.jsonl")  # ★ 变了
            os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
            with open(transcript_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"timestamp": time.time(), "message": msg},
                                   ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"完整日志追加失败：{e}")

    def _finalize(self, answer: str) -> str:
        """终态统一收口：写答案 → 摘要 → 刷盘 → 清理快照"""
        self._record_message({"role": "assistant", "content": answer})
        if hasattr(self.base_memory, "maybe_summary"):
            self.base_memory.maybe_summary()
        self.long_term_memory.flush()
        self._persist_history()

        # 终态后清理快照，避免残留旧状态
        if self.enable_checkpoint and os.path.exists(self.snapshot_path):
            try:
                os.remove(self.snapshot_path)
                logger.debug("任务完成，清理快照文件")
            except Exception as e:
                logger.warning(f"清理快照失败：{e}")

        return answer

    # ===================== 兜底执行 =====================
    def _react_fallback(
            self,
            query: str,
            extra_context: str = "",
            ltm_context: Optional[Dict] = None
    ) -> str:
        """ReAct 兜底：完整接入记忆 + 中间消息回写"""
        logger.info("触发 ReAct 优雅降级")
        if ltm_context is None:
            ltm_context = self.long_term_memory.build_memory_context(query)
        messages = self.base_memory.get_context().copy()
        if extra_context.strip():
            messages.append({
                "role": "system",
                "content": f"已完成的工作成果（无需重复执行）：\n{extra_context}",
            })
        if ltm_context:
            inserted = False
            for idx, msg in enumerate(messages):
                if msg["role"] == "system":
                    messages.insert(idx + 1, ltm_context)
                    inserted = True
                    break
            if not inserted:
                messages.insert(0, ltm_context)
        retrieval_text = "\n".join(m.get("content", "") for m in messages if m.get("content"))
        relevant_tools = self.tool_retriever.retriever(retrieval_text)
        tool_schema = build_tool_schema(relevant_tools)
        for _ in range(agent_conf["max_cycle"]):
            resp = self.llm_client.chat_completion(
                model=agent_conf["chat_model"],
                messages=messages,
                tool_schema=tool_schema,
            )
            msg = resp.choices[0].message
            messages.append(msg.model_dump())
            self._record_runtime(msg.model_dump())
            if not msg.tool_calls:
                return msg.content
            tool_results = run_tools_parallel(msg.tool_calls)
            for tc in msg.tool_calls:
                tool_msg = {"role": "tool", "tool_call_id": tc.id,
                            "content": tool_results[tc.id]}
                messages.append(tool_msg)
                self._record_runtime(tool_msg)
        return "抱歉，多次尝试后仍未得出结果，请换个方式提问。"

    # ===================== 新增：检查点持久化核心 =====================
    def _save_checkpoint(self) -> None:
        """
        原子写入检查点快照
        先写临时文件，写完再重命名，杜绝写一半崩溃损坏文件
        """
        if not self.enable_checkpoint:
            return
        if self._plan is None:
            return  # 还有生成计划，不需要保存

        try:
            snapshot = {
                "snapshot_version": SNAPSHOT_VERSION,
                "timestamp": time.time(),
                "agent_state": self._agent_state.value,
                "query": self._query,
                "plan_data": plan_to_dict(self._plan),
                "session_log": self._session_log.copy(),
                "memory_history": self.base_memory.history.copy(),  # ★ 新增：全量，断点续跑用
            }

            # 原子写入，同目录下写临时文件，成功后再rename
            dir_name = os.path.dirname(self.snapshot_path)
            os.makedirs(dir_name, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=dir_name,
                delete=False,
                suffix=".tmp"
            ) as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
                tmp_path = f.name

            os.replace(tmp_path, self.snapshot_path)
            logger.debug(f"检查点保存完成，状态：{self._agent_state.value}")

        except Exception as e:
            logger.warning(f"保存检查点失败：{e}，不影响主流程")

    def load_checkpoint(self) -> bool:
        """
        从快照文件恢复完整运行时状态
        返回 True 表示恢复成功，False 表示无快照或恢复失败
        """
        if not os.path.exists(self.snapshot_path):
            return False

        try:
            with open(self.snapshot_path, "r", encoding="utf-8") as f:
                snapshot = json.load(f)

            # 版本校验：版本不兼容直接放弃
            if snapshot.get("snapshot_version", 0) != SNAPSHOT_VERSION:
                logger.warning("快照版本不兼容，忽略")
                return False

            # 恢复基础字段
            self._agent_state = AgentState(snapshot["agent_state"])
            self._query = snapshot["query"]

            # 重建 Plan 对象
            self._plan = plan_from_dict(snapshot["plan_data"])

            # 恢复会话日志 + 注入短期记忆
            # session_log 照常恢复（_persist_history 要用）
            self._session_log = snapshot.get("session_log", [])

            # 优先用全量历史恢复 base_memory
            memory_history = snapshot.get("memory_history", [])
            if memory_history:
                self.base_memory.restore_history(memory_history)
            elif self._session_log:
                # 向后兼容：老快照没有 memory_history，退回 session_log
                self.base_memory.restore_history(self._session_log)

            # 重新召回长期记忆（向量库持久化，不需要存快照里）
            self._ltm_context = self.long_term_memory.build_memory_context(self._query)

            logger.info(
                f"快照恢复成功，当前状态：{self._agent_state.value}，"
                f"计划进度：{sum(1 for s in self._plan.steps if s.status == StepStatus.DONE)}/{len(self._plan.steps)}"
            )
            return True

        except Exception as e:
            logger.error(f"加载快照失败：{e}，将从头开始执行")
            return False

    # ===================== FSM 迁移入口 =====================
    def _transition(self, next_state: AgentState) -> None:
        """唯一状态变更入口：查迁移表 → 非法跳转抛异常 → 保存检查点"""
        allowed = _AGENT_TRANSITIONS.get(self._agent_state, set())
        if next_state not in allowed:
            raise ValueError(
                f"Agent状态非法跳转：{self._agent_state.value} → {next_state.value}"
            )
        old = self._agent_state
        self._agent_state = next_state
        logger.debug(f"[FSM] 状态变更：{old.value} → {next_state.value}")

        # ========== 新增：状态跳转完成，保存检查点 ==========
        self._save_checkpoint()

    # ===================== 各状态 handler =====================
    def _handle_init(self) -> AgentState:
        logger.info("[FSM] INIT 初始化")
        self._record_message({"role": "user", "content": self._query})
        self._ltm_context = self.long_term_memory.build_memory_context(self._query)
        return AgentState.PLANNING

    def _handle_planning(self) -> AgentState:
        logger.info("[FSM] PLANNING 规划中")
        plan_query = self._query
        if self._ltm_context:
            plan_query = (
                f"参考历史记忆：\n{self._ltm_context['content']}\n\n"
                f"用户当前需求：{self._query}"
            )
        self._plan = self.planner.generate_plan(plan_query)
        if self._plan.is_fallback:
            logger.warning("规划生成失败，进入失败终态")
            return AgentState.FAILED
        self._plan.transition("start_exec")
        self._print_plan(self._plan)
        return AgentState.EXECUTING

    def _handle_executing(self) -> AgentState:
        logger.info("[FSM] EXECUTING 步骤执行中")
        ready_steps = self._plan.get_ready_step()
        if not ready_steps:
            return AgentState.RESULT_PROCESSING
        for step in ready_steps:
            if step.status == StepStatus.PENDING:
                self.executor.execute_step(self._plan, step)
                if step.result:
                    self._record_runtime({
                        "role": "assistant",
                        "content": f"【步骤完成】{step.description}：{step.result}",
                    })
        return AgentState.RESULT_PROCESSING

    def _handle_result_processing(self) -> AgentState:
        logger.info("[FSM] RESULT_PROCESSING 结果处理中")
        if self._plan.all_done():
            logger.info("所有步骤完成，进入成功终态")
            return AgentState.COMPLETED
        failed_step = self._plan.get_failed_step()
        if failed_step:
            logger.info(f"检测到失败步骤：{failed_step.description}，进入反思")
            return AgentState.REFLECTING
        if self._plan.get_ready_step():
            return AgentState.EXECUTING
        logger.warning("流程异常：无失败、无待执行、未全部完成")
        return AgentState.FAILED

    def _handle_reflecting(self) -> AgentState:
        logger.info("[FSM] REFLECTING 反思重规划中")
        failed_step = self._plan.get_failed_step()
        if not failed_step:
            return AgentState.FAILED
        self._plan = self.reflector.replan(self._plan, failed_step)

        # ★ 加固：显式处理所有可能的返回状态，避免意外
        if self._plan.status == PlanStatus.FAILED:
            logger.info("重规划次数耗尽，进入失败终态")
            return AgentState.FAILED
        if self._plan.status == PlanStatus.EXECUTING:
            logger.info("重规划成功，返回执行")
            return AgentState.EXECUTING
        raise ValueError(f"重规划后出现意外状态：{self._plan.status.value}")

    # ===================== 终态处理 =====================
    def _handle_completed(self) -> None:
        logger.info("[FSM] COMPLETED 任务完成")
        self._plan.transition("complete")
        self._final_answer = _summarize_result(self.llm_client, self._plan)
        # ★ 新增：把答案回写到 Plan，保证序列化/日志里能读到完整终态
        self._plan.final_answer = self._final_answer

    def _handle_failed(self) -> None:
        logger.info("[FSM] FAILED 任务失败，启动优雅降级")
        if self._plan is None:
            self._final_answer = self._react_fallback(self._query)
            return
        done_steps = [
            f"{s.description}：{s.result}"
            for s in self._plan.steps
            if s.status == StepStatus.DONE and s.result
        ]
        extra_context = "\n".join(done_steps)
        self._final_answer = self._react_fallback(
            self._query, extra_context, self._ltm_context
        )

    # ===================== handler 分发表 =====================
    _HANDLERS = {
        AgentState.INIT: "_handle_init",
        AgentState.PLANNING: "_handle_planning",
        AgentState.EXECUTING: "_handle_executing",
        AgentState.RESULT_PROCESSING: "_handle_result_processing",
        AgentState.REFLECTING: "_handle_reflecting",
    }

    # ===================== 主入口 =====================
    def run(self, query: str, resume: bool = True) -> str:
        """
        Agent 主入口
        :param query: 用户问题
        :param resume: 是否尝试从断点恢复
        """
        self._query = query

        # ========== 新增：优先尝试断点恢复 ==========
        resumed = False
        if resume:
            resumed = self.load_checkpoint()

        # 没恢复成功，才从 INIT 开始
        if not resumed:
            self._agent_state = AgentState.INIT

        # 主循环：从当前状态继续流转
        while self._agent_state not in (AgentState.COMPLETED, AgentState.FAILED):
            handle_name = self._HANDLERS.get(self._agent_state)
            if handle_name is None:
                raise ValueError(f"没有为状态 {self._agent_state.value} 注册 handler")
            next_state = getattr(self, handle_name)()
            self._transition(next_state)

        # 终态副作用
        if self._agent_state == AgentState.COMPLETED:
            self._handle_completed()
        else:
            self._handle_failed()

        return self._finalize(self._final_answer)

    # ===================== 辅助方法 =====================
    @staticmethod
    def _print_plan(plan: Plan) -> None:
        print(f"\n===== 计划生成完成，共{len(plan.steps)}步 =====")
        for i, step in enumerate(plan.steps, 1):
            deps = [plan.get_step(d).description for d in step.depends_on]
            print(f"步骤{i}: {step.description}，依赖：{deps or '无'}")