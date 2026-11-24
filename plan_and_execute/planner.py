"""
Planner：根据用户需求生成结构化计划，不做执行，不做复盘
工业级加固：Schema契约校验LLM输出，非法输入在校验层拦截
"""

from plan_and_execute.plan_model import Plan, Step, PlanStatus, PlanContract
from client.llm_client import LLMClient
from utils.config_tool import agent_conf
from tools.tool_registry import tool_registry
from utils.logger_tool import logger


PLANNER_PROMPT = """
你是任务规划器。把用户需求拆成1-5个简单有序步骤，只输出合法JSON，不要解释、不要Markdown代码块。
格式：{"steps": [{"description": "步骤目标", "depends_on": [前置步骤序号，从0开始，无依赖填[]]}]}
可用工具：
"""

# 将工具注册表的信息拼接成大模型能看懂的字符串
def format_tools() -> str:
    return "\n".join(f"- {name}: {info['search_text']}" for name, info in tool_registry.items())

class Planner:
    def __init__(self, llm_client: LLMClient, max_plan_steps: int = 5):
        self.client = llm_client
        self.model = agent_conf["chat_model"]
        self.max_plan_steps = max_plan_steps

    def _fall_back(self, query: str) -> Plan:
        """兜底单步计划，明确标记is_fallback，不用字符串猜"""
        return Plan(
            goal=query,
            steps=[Step(description=query)],
            is_fallback=True,
        )

    def generate_plan(self, query: str) -> Plan:
        """调用大模型生成结构性计划，解析JSON"""
        try:
            # 构建 Prompt（和原来完全一致）
            messages = [
                {"role": "system", "content": PLANNER_PROMPT + format_tools()},
                {"role": "user", "content": query}
            ]

            # ========== 核心新增：原生结构化输出约束 ==========
            # 从 Pydantic 契约自动生成 JSON Schema，传给接口做生成时约束
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "plan_output",
                    "schema": PlanContract.model_json_schema()
                }
            }

            # 1. 调用 LLM，加原生格式约束
            response = self.client.chat_completion(
                model=self.model,
                messages=messages,
                tool_schema=[],
                temperature=0.1,
                response_format=response_format  # 新增这一行
            )
            raw_text = response.choices[0].message.content
            logger.debug(f"Planner原始输出：{raw_text}")

            # Pydantic 二次校验（纵深防御）
            # 不在需要正则清洗，parsed_dict = clean_llm_json(raw_text)

            # ========== 核心新增：Schema契约强校验 ==========
            # 3. 用契约校验所有字段、类型、依赖合法性
            # 类型错、字数超、依赖越界、自依赖，这里都会直接抛异常
            contract = PlanContract.model_validate_json(raw_text)
            logger.info(f"计划契约校验通过，共{len(contract.steps)}步")

            # 4. 契约 → 运行时模型转换（和原来的逻辑完全一致）
            # 第一轮：创建所有Step对象，生成唯一step_id
            steps = [Step(description=s.description) for s in contract.steps]
            idx_to_id = {i: s.step_id for i, s in enumerate(steps)}

            # 第二轮：回填依赖，把序号替换为真实的step_id
            for i, s in enumerate(steps):
                s.depends_on = [
                    idx_to_id[d]
                    for d in contract.steps[i].depends_on
                    if d in idx_to_id
                ]
            return Plan(goal=query, steps=steps, status=PlanStatus.INITIALIZED)
        except Exception as e:
            logger.info(f"Planner生成失败：{e}，降级兜底计划")
            return self._fall_back(query)
