"""
Executor：只负责执行单个Step，内部跑ReAct小循环，不做全局规划
状态机改造：所有状态变更通过 transition 事件驱动，禁止直接赋值 status
"""
from plan_and_execute.plan_model import Plan, Step, StepStatus
from client.llm_client import LLMClient
from tools.tool_retriever import ToolRetriever
from pydantic_schemas.build_tool_schema import build_tool_schema
from tools.run_tools_parallel import run_tools_parallel
from utils.config_tool import agent_conf
from utils.logger_tool import logger


class Executor:
    def __init__(
            self,
            llm_client: LLMClient,
            tool_retriever: ToolRetriever,
            max_inner_round: int = 3,
            max_retry: int = 2,
    ):
        self.client = llm_client
        self.model = agent_conf["chat_model"]
        self.tool_retriever = tool_retriever
        self.max_inner_round = max_inner_round
        self.max_retry = max_retry

    def _build_messages(self, plan: Plan, step: Step) -> list[dict]:
        """
        拼接单个Step的上下文：system prompt + 依赖步骤信息 + 当前步骤的目标
        计划一旦复杂（5 步 + 每步结果 1000 字 = 5K 字），传给 LLM 的上下文会指数膨胀，而且大部分是当前步骤用不到的
        设计原则：前置约束越明确，模型无效调用越少，单步收敛速度越快
        """
        # ★ 全局 system prompt，统一身份设定
        msgs = [{"role": "system", "content": agent_conf["system_prompt"]}]

        # 只传当前步骤真正依赖的，避免无关信息干扰，减少token浪费
        dep_results = []
        for dep_id in step.depends_on:
            dep = plan.get_step(dep_id)
            if dep and dep.status == StepStatus.DONE and dep.result:
                dep_results.append(f"【依赖结果 - {dep.description}】\n{dep.result}")
        if dep_results:
            msgs.append({
                "role": "system",
                # 强调“真实有效、可直接引用、无需重复验证”，消除模型的验证焦虑
                "content": "以下是已经完成的前置步骤结果，数据真实有效，可直接引用，无需重复验证：\n\n" + "\n\n".join(
                    dep_results)
            })

            # 3. 单步执行核心约束（解决重复调用的关键！）
            # 明确4件事：目标是什么、完成标准是什么、禁止什么行为、输出格式要求
        step_constraint = f"""
        你当前只需要完成下面这一个步骤，不要做额外事情：
        【步骤目标】{step.description}

        【严格执行规则】
        1.  结果复用：同一步内已经计算出的数值、前置步骤给出的数值，直接引用即可，禁止重复计算相同的算式。
        2.  完成标准：当步骤目标要求的所有数值都计算完成、所有结论都得出后，直接输出最终结果，不需要再调用工具验算、交叉验证。
        3.  禁止冗余：相同参数的工具只需要调用一次，禁止无意义的重复调用、拆分重算、反复验算。
        4.  输出格式：最终直接输出结构化的步骤结论，分点清晰，不要罗列工具调用过程，不要解释计算思路。

        请开始执行这一步。
        """
        # 去除首尾空行，保持提示词整洁
        msgs.append({"role": "user", "content": step_constraint.strip()})
        return msgs

    def execute_step(self, plan: Plan, step: Step) -> Step:
        """执行单个Step（状态机全事件驱动）"""
        # 【改动点】启动步骤：PENDING → IN_PROGRESS
        step.transition("start")
        logger.info(f"\n===== 执行步骤：{step.description}（第{step.retry_count + 1}次尝试） =====")

        # 获取工具列表，创建工具schema
        tools = self.tool_retriever.retriever(step.description)
        tool_schema = build_tool_schema(tools)
        messages = self._build_messages(plan, step)

        # 内层ReAct循环
        for i in range(self.max_inner_round):
            try:
                response = self.client.chat_completion(
                    model=self.model,
                    messages=messages,
                    tool_schema=tool_schema,
                    temperature=0,
                )
                assistant_msg = response.choices[0].message

                # 没有工具调用：步骤完成
                if not assistant_msg.tool_calls:
                    # 【改动点】步骤成功：IN_PROGRESS → DONE
                    step.transition("succeed")
                    step.result = assistant_msg.content
                    logger.info(f"✅ 完成：{step.result[:100]}...")
                    return step

                # 有工具调用：执行后把结果回传，进入下一轮
                messages.append(assistant_msg.model_dump())
                logger.info(f"🔧 调用工具：{[tc.function.name for tc in assistant_msg.tool_calls]}")
                # 执行工具，得到结果
                tool_results = run_tools_parallel(assistant_msg.tool_calls)
                for tc in assistant_msg.tool_calls:
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_results[tc.id]})

            except Exception as e:
                step.retry_count += 1
                step.error = f"第{i+1}轮异常：{e}"
                return self._after_try(step)

        # 内循环耗尽也走重试逻辑
        step.retry_count += 1
        step.error = f"内循环{self.max_inner_round}轮未完成"
        return self._after_try(step)

    def _after_try(self, step: Step) -> Step:
        """
        统一处理重试/失败（严格遵循状态机迁移规则）
        流程：先标记失败 → 未达上限则重置回待执行 → 达上限则终态失败
        """
        # 第一步：统一标记失败（IN_PROGRESS → FAILED）
        step.transition("fail", error_msg=step.error)

        # 第二步：判断是否可以重试
        if step.retry_count < self.max_retry:
            # 可以重试：FAILED → PENDING，回到调度队列
            step.transition("reset")
            logger.info(f"🔄 等待重试（{step.retry_count}/{self.max_retry}）")
        else:
            logger.info(f"❌ 步骤失败：{step.error}")

        return step
