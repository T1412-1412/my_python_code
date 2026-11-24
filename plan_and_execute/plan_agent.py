"""
规划层agent主流程：线性串联Planner/Executor/Reflector，带ReAct兜底
状态机改造：计划宏观状态通过 transition 事件驱动
"""
from plan_and_execute.plan_model import Plan, PlanStatus, StepStatus
from plan_and_execute.planner import Planner
from plan_and_execute.executor import Executor
from plan_and_execute.reflector import Reflector
from client.llm_client import LLMClient
from tools.tool_retriever import ToolRetriever
from pydantic_schemas.build_tool_schema import build_tool_schema
from tools.run_tools_parallel import run_tools_parallel
from utils.config_tool import agent_conf
from utils.logger_tool import logger


def _react_fallback(
    llm: LLMClient,
    retriever: ToolRetriever,
    query: str,
    completed_context: str = "",
) -> str:
    """
    规划层失败兜底：两种情况
    1. 计划生成失败，completed_context为空，直接走纯ReAct
    2. 步骤执行过程出现失败，completed_context非空，带着已完成结果继续，不从头开始
    """
    logger.info("🔻 降级到ReAct模式" + ("（携带已完成上下文）" if completed_context else ""))
    if completed_context:
        user_content = (
            f"原始需求：{query}\n\n"
            f"已经完成的部分（无需重复）：\n{completed_context}\n\n"
            f"请基于以上已完成信息，继续完成剩余部分。"
        )
    else:
        user_content = query
    msgs = [{"role": "user", "content": user_content}]
    schema = build_tool_schema(retriever.retriever(query))
    for _ in range(agent_conf["max_cycle"]):
        resp = llm.chat_completion(
            model=agent_conf["chat_model"], messages=msgs, tool_schema=schema
        )
        msg = resp.choices[0].message
        msgs.append(msg.model_dump())
        if not msg.tool_calls:
            return msg.content
        results = run_tools_parallel(msg.tool_calls)
        for tc in msg.tool_calls:
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": results[tc.id]})
    return "抱歉，多次尝试后仍未得出结果，请换个方式提问。"


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
            reflector: Reflector
    ):
        self.client = llm_client
        self.retriever = retriever
        self.planner = planner
        self.executor = executor
        self.reflector = reflector

    def run(self, query: str) -> str:
        # 1. 生成计划，计划生成失败直接走兜底
        plan = self.planner.generate_plan(query)
        # 调用点 1：Planner 失败 → 纯 ReAct，没有已完成步骤，不传 context
        if plan.is_fallback:
            return _react_fallback(self.client, self.retriever, query)

        # 【改动点】启动计划：INITIALIZED → EXECUTING
        plan.transition("start_exec")
        self._print_plan(plan)

        # 2. 外层执行循环
        while True:
            # 有失败步骤且重试耗尽，进行重规划
            failed_steps = plan.get_failed_step()
            if failed_steps:
                # 将重规划后的计划赋值给原计划
                plan = self.reflector.replan(plan, failed_steps)
                # 判断新生成的计划是否失败，失败的话进行兜底
                if plan.status == PlanStatus.FAILED:
                    # 调用点 2：重规划耗尽 → 优雅降级，带已完成上下文
                    return self._graceful_degrade(plan, query)
                continue

            # 获取可执行的步骤
            pending_steps = plan.get_ready_step()
            # 没有可执行步骤退出循环
            if not pending_steps:
                break

            # 串行执行所有可执行步骤
            for s in pending_steps:
                if s.status == StepStatus.PENDING:
                    self.executor.execute_step(plan, s)

        # 3. 全部步骤完成，进行汇总，失败进行兜底
        if plan.all_done():
            # 【改动点】计划完成：EXECUTING → COMPLETED
            plan.transition("complete")
            plan.final_answer = _summarize(self.client, plan)
            return plan.final_answer

        # 调用点 3：仍有未完成 → 优雅降级，带已完成上下文
        return self._graceful_degrade(plan, query)

    def _graceful_degrade(self, plan: Plan, query: str) -> str:
        """部分完成时兜底：有已完成结果就带着走，没有才纯 ReAct"""
        done = [
            f"{s.description}：{s.result}"
            for s in plan.steps
            if s.status == StepStatus.DONE and s.result
        ]
        context = "\n".join(done)
        return _react_fallback(self.client, self.retriever, query, context)

    @staticmethod
    def _print_plan(plan: Plan):
        print(f"\n===== 计划完成，共{len(plan.steps)}步 =====")
        for i, s in enumerate(plan.steps, 1):
            deps = [plan.get_step(d).description for d in s.depends_on]
            print(f"步骤{i}: {s.description}，依赖：{deps or '无'}")
