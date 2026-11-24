"""
ReAct agent_code：实现智能体思考 -> 执行 -> 拿到结果 -> 再思考 -> 直到得到最终结果
本质上是while循环和条件判断，重点是判断何时结束循环
"""

import os
import requests
from openai import OpenAI
import re
import json

def get_weather(city: str):
    """
    实现天气查询的功能，输入城市名称，返回天气信息
    :param city: 城市名称字符串
    :return: 格式化天气信息字符串
    """
    url = f"https://wttr.in/{city}?format=j1"

    response = requests.get(url)

    weather_data = response.json()

    return json.dumps(
        {
            "城市": city,
            "天气信息": weather_data["current_condition"][0],
            "温度": f"{weather_data['current_condition'][0]['temp_C']}°C",
            "风速": f"{weather_data['current_condition'][0]['windspeedKmph']} km/h",
            "湿度": f"{weather_data['current_condition'][0]['humidity']}%"
        },
        ensure_ascii=False
    )

def calculate_add(a: float, b: float):
    """
    计算两个数的和
    :param a: 第一个数
    :param b: 第二个数
    :return: 两个数的和
    """
    return a+b

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_add",
            "description": "计算两个数的和",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个数"},
                    "b": {"type": "number", "description": "第二个数"}
                },
                "required": ["a", "b"]
            }
        }
    }
]

# 工具调度字典
TOOL_MAP = {
    "get_weather": get_weather,
    "calculate_add": calculate_add
}

# 工具执行函数
def execute_tool(tool_name: str, tool_args: dict):
    if tool_name not in TOOL_MAP:
        return f"错误：未知工具{tool_name}"

    tool_func = TOOL_MAP[tool_name]
    try:
        return tool_func(**tool_args)
    except Exception as e:
        return f"工具执行出错：{str(e)}"

# 工具描述文本（给大模型看的工具清单）
TOOL_DESCRIPTION = """
可用的工具列表：
1. get_weather：获取指定城市的天气信息
    输入参数：city（字符串），表示城市的名称，例如：北京、上海
2. calculate_add：计算两个数的和
    输入参数：a（数字），表示第一个数；b（数字），表示第二个数
"""

# ReAct系统提示词
system_prompt = f"""你是一个擅长推理和使用工具的智能助手。你必须严格按照以下格式进行思考和行动。

{TOOL_DESCRIPTION}

## 执行规则
1. 你必须遵循[ Thought -> Action -> Action Input ]的格式进行输出，每一步只做一个行动。
2. 当你获取足够的信息后，输出 Final Answer 作为最终答案，结束任务。
3. 不要编造信息，不知道的内容必须调用工具查询。
4. 严格遵守格式，不要额外解释，不要输出多余内容。

## 输出格式标准
思考过程：
Thought: <你对当前问题的分析和下一步计划>
Action: <你要调用的工具名称，必须从可用工具列表中选择>
Action Input: <工具的输入参数，JSON 格式，例如 {{"city": "北京"}}>

得到工具返回结果后，继续思考，直到可以给出答案。
最终答案格式：
Final Answer: <你的最终回答>

## 示例
用户问题：北京今天多少度？
Thought: 用户想知道北京的温度，我需要调用天气查询工具。
Action: get_weather
Action Input: {{"city": "北京"}}
（得到观察结果后继续）
Thought: 我已经获取到北京的温度信息，可以回答用户了。
Final Answer: 北京当前温度为 25℃，天气晴朗。
"""

# 输出解析器
# 大模型返回的是自然语言，我们需要用正则从中提取出工具名和参数，或者判断是否输出了最终答案。
def parse_agent_output(output: str):
    """
    解析大模型的 ReAct 格式输出
    返回：
    - 有最终答案：{"type": "answer", "content": "答案内容"}
    - 要调用工具：{"type": "action", "tool": "工具名", "args": {参数字典}}
    - 解析失败：{"type": "error", "content": "错误信息"}
    :param output:大模型返回的文本字符串
    :return:
    """
    # 1. 先判断是否有最终答案
    match_final_answer = re.search(r"Final Answer:\s*(.+)", output, re.DOTALL)
    if match_final_answer:
        return {"type": "answer", "content": match_final_answer.group(1)}

    # 2. 提取Action（工具名）
    match_action = re.search(r"Action:\s*(\w+)", output)

    # 3. 提取Action Input（工具参数 JSON格式）
    match_action_input = re.search(r"Action Input:\s*(\{.*?\})", output, re.DOTALL)

    if match_action and match_action_input:
        tool_name = match_action.group(1).strip()
        try:
            tool_args = json.loads(match_action_input.group(1).strip())
            return {
                "type": "action",
                "tool": tool_name,
                "args": tool_args
            }
        except json.JSONDecodeError:
            return {"type": "error", "content": "参数格式解析失败，不是合法 JSON"}

    return {"type": "error", "content": "无法识别输出格式"}

# ReAct Agent主循环
class ReActAgent:
    def __init__(self, api_key: str, base_url: str, model: str, max_step: int = 5):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_step = max_step    # 最大循环步数，防止死循环
        self.messages = [{"role": "system", "content": system_prompt}]

    def run(self, user_query):
        self.messages.append({"role": "user", "content": user_query})

        for step in range(1, self.max_step+1):
            print(f"\n==========第{step}步思考过程============")
            # 调用大模型进行思考
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=0,
                # tools=tools,
                # tool_choice="auto"
            )
            llm_output = response.choices[0].message.content
            print(f"大模型输出：\n{llm_output}")

            # 解析大模型输出
            parse_result = parse_agent_output(llm_output)

            # 情况1：得出最终答案，循环结束
            if parse_result["type"] == "answer":
                print("得到最终答案")
                return parse_result["content"]

            # 情况2：解析失败，提示大模型修正
            elif parse_result["type"] == "error":
                print("解析失败，提示大模型修正")
                error_message = f"格式错误：{parse_result['content']}，请严格按照指定格式输出。"
                self.messages.append({"role": "assistant", "content": llm_output})
                self.messages.append({"role": "user", "content": error_message})
                continue

            # 情况3：需要调用工具
            elif parse_result["type"] == "action":
                print("需要调用工具")
                tool_name = parse_result["tool"]
                tool_args = parse_result["args"]

                if tool_name not in TOOL_MAP:
                    observation = f"错误：未知工具{tool_name}"
                else:
                    observation = execute_tool(tool_name, tool_args)

                # 将工具调用结果添加到消息列表中
                self.messages.append({"role": "assistant", "content": llm_output})
                self.messages.append({"role": "user", "content": f"Observation: {observation}"})

        # 循环达到最大步数，仍未得到答案
        return "抱歉，思考步数已到上限，未能解决问题。"




if __name__ == '__main__':
    api_key = os.getenv("API_KEY_qwen")
    base_url = "https://ws-75ntouay5nt8xgfs.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    agent = ReActAgent(api_key=api_key, base_url=base_url, model="qwen3.7-max")
    res = agent.run("北京的天气怎么样，出门要带伞吗，穿什么衣服啊")
    print(res)