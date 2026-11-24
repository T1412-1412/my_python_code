"""
规划层agent主流程：记忆 + 规划 + 工具 + 状态机 完整内核
修复点：
1. 所有兜底路径统一落盘，不会丢失对话
2. 兜底路径接入完整记忆体系（长期+短期+已完成结果），不再失忆
"""
from plan_and_execute.plan_model import Plan, PlanStatus, StepStatus
from plan_and_execute.planner import Planner
from plan_and_execute.executor import Executor
from plan_and_execute.reflector import Reflector
from client.llm_client import LLMClient
from tools.tool_retriever import ToolRetriever
from pydantic_schemas.build_tool_schema import build_tool_schema
from tools.run_tools_parallel import run_tools_parallel
from memory.base_memory import BaseMemory
from memory.long_term_memory import LongTermMemory
from utils.config_tool import agent_conf
from utils.logger_tool import logger
from typing import List, Dict
import os
import json
import time
from utils.path_tool import get_abs_path


def _summarize(llm: LLMClient, plan: Plan) -> str:
    """汇总各步结果，调用失败直接拼接兜底"""
    done = [f"{s.description}: {s.result}" for s in plan.steps if s.status == StepStatus.DONE]
    try:
        response = llm.chat_completion(
            model=agent_conf["chat_model"],
            messages=[
                {"role": "system", "content": "把下面步骤结果整合成通顺完整的回答，不要罗列步骤"},
                {"role": "user", "content": f"需求：{plan.goal}\n" + "\n".join(done)}
            ],
            tool_schema=[],
        )
        return response.choices[0].message.content
    except Exception as e:
        # 汇总失败直接拼接，保证有输出
        logger.warning(f"汇总失败，降级拼接：{e}")
        return "\n".join(done)


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
    ):
        self.client = llm_client
        self.retriever = retriever
        self.planner = planner
        self.executor = executor
        self.reflector = reflector
        # 记忆体系
        self.base_memory = base_memory
        self.long_term_memory = long_term_memory
        # 完整会话日志，用于持久化
        self._session_log: List[Dict] = []
        # 幂等标志：避免重复加载磁盘历史、重复注入
        self._history_inited = False
        # 初始化时加载历史
        self._init_history_from_disk()

    def _init_history_from_disk(self) -> None:
        """
        从磁盘加载完整历史，并同步到 base_memory 和 _session_log。
        只执行一次（幂等），避免重复加载、重复注入 system。
        """
        if self._history_inited:
            return

        state_path = get_abs_path("history_messages/memory_state.json")
        # 1) 优先用快照恢复（含摘要状态）
        restored = False
        if hasattr(self.base_memory, "load_state"):
            restored = self.base_memory.load_state(state_path)

        # 2) 快照不可用，退回原有逻辑
        if not restored:
            history = self._load_history()
            if history:
                # 1) 完整历史 → _session_log，用于持久化时"全量覆盖"
                self._session_log = history
                # 2) 注入 base_memory，用于当次对话的上下文构建
                self.base_memory.restore_history(history)
                logger.info(f"历史记忆注入完成，共 {len(history)} 条")

        # 3) session_log 始终维护一份（后续持久化、审计都用得到）
        if not self._session_log and restored:
            # 用 _history 的副本重建（此时 _history 已经是最新的权威数据）
            self._session_log = self.base_memory.history

        self._history_inited = True

    def _record_runtime(self, msg: Dict) -> None:
        """
        运行时消息：ReAct 内循环、步骤中间结果
        只进 L1 当次上下文 + L3 完整日志，不进 L2 对话历史、不进 L4 长期记忆
        """
        self.base_memory.add_message(msg)  # L1
        self._append_transcript(msg)  # L3
        # 不进 self._session_log（L2 对话历史）
        # 不进 self.long_term_memory（L4 长期记忆）

    def _append_transcript(self, msg: Dict) -> None:
        """追加写入完整对话日志（JSONL 格式，append-only）"""
        try:
            transcript_path = get_abs_path("history_messages/transcript.jsonl")
            os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
            # ★ 关键：用 "a" 追加，不是 "w" 覆盖
            with open(transcript_path, "a", encoding="utf-8") as f:
                record = {
                    "timestamp": time.time(),
                    "message": msg,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"完整日志追加失败：{e}")

    def _record(self, msg: Dict) -> None:
        """
        消息统一写入入口：四层全写
        短期记忆(上下文) + 长期记忆(向量库缓冲) + 会话日志(持久化)
        和 _008agent 逻辑完全对齐
        """
        self.base_memory.add_message(msg)  # L1
        self._session_log.append(msg)  # L2（用于 _save_history）
        self._append_transcript(msg)  # L3
        self.long_term_memory.add_message(msg)  # L4

    def _load_history(self):
        file_path = get_abs_path("history_messages/history_messages.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    messages = json.load(f)
                # if isinstance(messages, list) and len(messages) > 0:
                #     # 磁盘存储第一条永远是system消息，原地pop删除，不要赋值接收pop返回值
                #     messages.pop(0)
                    logger.debug("历史记录加载成功")
                    return messages
            except Exception as e:
                logger.error(f"错误：加载历史记录失败，错误信息 {str(e)}")
                return []
        logger.debug("历史记录不存在")
        return []

    def _save_history(self, messages: list[dict]):
        """持久化会话历史，和 _008agent 格式完全兼容"""
        file_path = get_abs_path("history_messages/history_messages.json")
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False)
            logger.info("会话记录保存成功")
        except Exception as e:
            logger.error(f"保存历史记录失败：{str(e)}")

        # 摘要记忆快照持久化（如果支持）
        if hasattr(self.base_memory, "save_state"):
            state_path = get_abs_path("history_messages/memory_state.json")
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            self.base_memory.save_state(state_path)

    def _finalize(self, answer: str) -> str:
        """
        【统一收尾出口】所有正常/兜底路径最终都走这里
        保证：写入助手消息 → 触发摘要 → 长期记忆落盘 → 会话历史持久化
        永远不会出现「对话没存盘」的情况
        """
        self._record({"role": "assistant", "content": answer})
        # 触发摘要压缩（如果支持）
        if hasattr(self.base_memory, "maybe_summary"):
            self.base_memory.maybe_summary()
        # 长期记忆强制落盘
        self.long_term_memory.flush()
        # 会话历史持久化到磁盘
        self._save_history(self._session_log)
        return answer

    def _react_fallback(self, query: str, extra_context: str = "", ltm_context: dict = None) -> str:
        """
        ReAct兜底：接入完整记忆体系，不再失忆
        上下文构成：长期记忆召回 + 短期历史上下文 + 额外补充信息（已完成步骤） + 用户需求
        和 _008agent 的主循环逻辑完全对齐
        """
        logger.info("🔻 降级到ReAct模式")

        # 1. 召回长期记忆（和主流程完全一致）
        if ltm_context is None:
            ltm_context = self.long_term_memory.build_memory_context(query)

        # 2. 取短期记忆已有上下文（当前会话已经产生的所有历史）
        base_context = self.base_memory.get_context()
        messages = base_context.copy()

        # 3. 拼接额外补充信息（已完成步骤结果）
        if extra_context.strip():
            messages.append({
                "role": "system",
                "content": f"已完成的工作成果（无需重复执行）：\n{extra_context}"
            })

        # 4. 补充长期记忆
        if ltm_context:
            # 插在system之后，保证模型优先看到
            inserted = False
            for idx, msg in enumerate(messages):
                if msg["role"] == "system":
                    messages.insert(idx + 1, ltm_context)
                    inserted = True
                    break
            if not inserted:
                messages.insert(0, ltm_context)

        # 5. 工具检索：用完整上下文检索，更精准
        retrieval_text = "\n".join([m.get("content", "") for m in messages if m.get("content")])
        relevant_tools = self.retriever.retriever(retrieval_text)
        tool_schema = build_tool_schema(relevant_tools)

        # 6. ReAct主循环（和 _008agent 逻辑完全一致）
        for _ in range(agent_conf["max_cycle"]):
            resp = self.client.chat_completion(
                model=agent_conf["chat_model"],
                messages=messages,
                tool_schema=tool_schema
            )
            msg = resp.choices[0].message
            messages.append(msg.model_dump())
            self._record_runtime(msg)  # ★ 修复新回归

            if not msg.tool_calls:
                return msg.content

            tool_results = run_tools_parallel(msg.tool_calls)
            for tc in msg.tool_calls:
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_results[tc.id]
                }
                messages.append(tool_msg)
                self._record_runtime(tool_msg)  # ★ 修复新回归

        return "抱歉，多次尝试后仍未得出结果，请换个方式提问。"

    def _graceful_degrade(self, plan: Plan, query: str, ltm_context: dict = None) -> str:
        """
        优雅降级：带着已完成步骤的结果走ReAct，不推倒重来
        只负责提取已完成成果，真正的兜底执行交给 _react_fallback
        """
        if ltm_context is None:
            ltm_context = self.long_term_memory.build_memory_context(query)
        done = [
            f"{s.description}：{s.result}"
            for s in plan.steps
            if s.status == StepStatus.DONE and s.result
        ]
        extra_context = "\n".join(done)
        logger.info(f"优雅降级，已完成{len(done)}步，带着成果继续执行")
        return self._react_fallback(query, extra_context, ltm_context)

    def run(self, query: str) -> str:
        # ========== 1. 记忆前置：写入用户消息 + 召回长期记忆 ==========
        self._record({"role": "user", "content": query})
        ltm_context = self.long_term_memory.build_memory_context(query)

        # ========== 2. 生成计划 ==========
        # 把长期记忆拼进规划上下文，让Planner基于历史信息规划
        plan_query = query
        if ltm_context:
            plan_query = f"参考历史记忆：\n{ltm_context['content']}\n\n用户当前需求：{query}"

        plan = self.planner.generate_plan(plan_query)

        # 兜底入口1：Planner生成失败 → 直接走ReAct
        if plan.is_fallback:
            logger.info("计划生成失败，直接降级ReAct")
            answer = self._react_fallback(query)
            return self._finalize(answer)

        # 启动计划：INITIALIZED → EXECUTING
        plan.transition("start_exec")
        self._print_plan(plan)

        # ========== 3. 外层执行循环 ==========
        while True:
            # 优先处理失败步骤 → 重规划
            failed_step = plan.get_failed_step()
            if failed_step:
                plan = self.reflector.replan(plan, failed_step)
                # 兜底入口2：重规划耗尽 → 优雅降级
                if plan.status == PlanStatus.FAILED:
                    answer = self._graceful_degrade(plan, query)
                    return self._finalize(answer)
                continue

            # 获取可执行步骤
            pending_steps = plan.get_ready_step()
            if not pending_steps:
                break

            # 串行执行所有可执行步骤
            for step in pending_steps:
                if step.status == StepStatus.PENDING:
                    self.executor.execute_step(plan, step)
                    # 步骤结果同步写入会话记忆
                    if step.result:
                        self._record_runtime({  # ← 从 _record 改为 _record_runtime
                            "role": "assistant",
                            "content": f"【步骤完成】{step.description}：{step.result}"
                        })

        # ========== 4. 结果处理 ==========
        if plan.all_done():
            plan.transition("complete")
            final_answer = _summarize(self.client, plan)
        else:
            # 兜底入口3：步骤未全部完成 → 优雅降级
            final_answer = self._graceful_degrade(plan, query)

        # 正常流程统一收尾
        return self._finalize(final_answer)


    @staticmethod
    def _print_plan(plan: Plan):
        print(f"\n===== 计划完成，共{len(plan.steps)}步 =====")
        for i, s in enumerate(plan.steps, 1):
            deps = [plan.get_step(d).description for d in s.depends_on]
            print(f"步骤{i}: {s.description}，依赖：{deps or '无'}")
