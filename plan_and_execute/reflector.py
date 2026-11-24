"""
Reflector：只负责失败后复盘、修正剩余计划，不执行任务
状态机改造：计划终态失败通过 transition 事件驱动，重规划中保持EXECUTING状态
"""
from plan_and_execute.plan_model import Step, Plan, StepStatus, ReplanningContract
from client.llm_client import LLMClient
from utils.config_tool import agent_conf
from utils.logger_tool import logger


REFLECTOR_PROMPT = """
你是复盘重规划专家。基于已完成的结果和失败原因，修正剩余步骤，不要重复已完成内容。
严格输出JSON，不要任何解释、不要Markdown代码块。

输出格式：
{{
  "new_remaining_steps": [
    {{
      "description": "修正后的步骤描述",
      "depends_on_index": [剩余步骤内部的前置序号，从0开始，无依赖填[]]
    }}
  ],
  "reason": "本次重规划的原因说明"
}}

规则：
1. 剩余步骤最多5个
2. depends_on_index只填剩余步骤内部的序号，不要引用已完成的步骤
3. 无法完成则new_remaining_steps返回空数组
"""


class Reflector:
    def __init__(self, llm_client: LLMClient, max_replan: int = 2):
        self.client = llm_client
        self.model = agent_conf["chat_model"]
        self.max_replan = max_replan

    def replan(self, plan: Plan, failed_step: Step) -> Plan:
        """执行重规划（状态机：EXECUTING → REPLANNING → EXECUTING/FAILED）"""
        # 重规划次数是否到达上限
        if plan.is_replan_exhausted(self.max_replan):
            logger.info(f"达到最大重规划次数：{self.max_replan}，重规划失败")
            # 【改动点】计划彻底失败：EXECUTING → FAILED
            plan.transition("fail")
            return plan

        # ★ 进入重规划状态
        plan.transition("start_replan")  # EXECUTING → REPLANNING

        try:
            # 1. 拼接上下文
            done_steps = [f"{s.description}：{s.result}" for s in plan.steps if s.status == StepStatus.DONE]
            messages = [
                {"role": "system", "content": REFLECTOR_PROMPT},
                {"role": "user",
                 "content": f"原始目标：{plan.goal}\n已完成：{done_steps or '无'}\n失败步骤：{failed_step.description}\n原因：{failed_step.error}"}
            ]

            # 原生结构化输出约束
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "replanning_output",
                    "schema": ReplanningContract.model_json_schema()
                }
            }

            # 2. 调用 LLM
            response = self.client.chat_completion(
                model=self.model,
                messages=messages,
                tool_schema=[],
                temperature=0.1,
                response_format=response_format
            )
            raw_text = response.choices[0].message.content
            logger.debug(f"Reflector原生输出：{raw_text}")

            # 3. 契约校验
            contract = ReplanningContract.model_validate_json(raw_text)
            logger.info(f"重规划契约校验通过，剩余{len(contract.new_remaining_steps)}步，原因：{contract.reason}")

            # 空步骤 = LLM 明确判定无法完成，计划应终态失败
            if not contract.new_remaining_steps:
                logger.info("重规划判定无法继续，计划置为 FAILED")
                plan.transition("fail")
                return plan

            # 4. 保留已完成步骤
            kept_steps = [s for s in plan.steps if s.status == StepStatus.DONE]
            new_steps = [Step(description=s.description) for s in contract.new_remaining_steps]

            # 5. 回填新步骤内部依赖
            idx_to_id = {i: s.step_id for i, s in enumerate(new_steps)}
            for i, step in enumerate(new_steps):
                internal_deps = [
                    idx_to_id[d]
                    for d in contract.new_remaining_steps[i].depends_on_index
                    if d in idx_to_id
                ]
                # 锚点依赖：保证新步骤整体承接在已完成成果之后
                if kept_steps:
                    step.depends_on = [kept_steps[-1].step_id] + internal_deps
                else:
                    step.depends_on = internal_deps

            # 6. 更新计划 + 计数（计划状态保持EXECUTING，无需重复赋值）
            plan.steps = kept_steps + new_steps
            plan.replan_count += 1
            logger.info(f"🔄 第{plan.replan_count}次重规划完成，剩余{len(new_steps)}步")

            # ★ 重规划成功，回到执行态
            plan.transition("replan_ok")  # REPLANNING → EXECUTING

            return plan

        except Exception as e:
            logger.info(f"重规划失败：{e}")
            # ★ 注意事件名是 replan_fail，不是 fail（当前状态是 REPLANNING，不是 EXECUTING）
            plan.transition("replan_fail")  # REPLANNING → FAILED
            return plan
