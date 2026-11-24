"""
参数校验解决的是[输入不对]的问题，在真实场景中，工具本身也可能失败，
比如网络请求失败，第三方api出问题，服务限流等等。
为了解决这些问题，就需要执行层的容错重试，我们可以使用工具重试机制。
## 核心逻辑
区分两种错误：

- 参数错误：让模型自己修正（自纠错闭环）
- 执行错误：程序自动重试，重试失败再告诉模型
"""

import os
import json
import time

import requests
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
from utils.path_tool import get_abs_path

# 工具调用失败最大重试次数，防止死循环，默认为3次
MAX_RETRY = 3

# 天气工具函数
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息
    :param city: 城市名称
    :return: 天气信息
    """
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=30)
        weather_data = response.json()
        return json.dumps(weather_data, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"天气查询失败：{str(e)}"}, ensure_ascii=False)

# 计算器工具函数
def calculator(a: float, b: float, operator: str) -> float|str:
    """
    简易的计算器，计算两个数的和、差、积、商
    :param a: 数字1
    :param b: 数字2
    :param operator: 运算符必须为其中之一：+、-、*、/
    :return: 计算结果，如果运算符无效则返回错误信息
    """
    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        return a * b
    elif operator == "/":
        if b != 0:
            return round(a / b, 2)
        else:
            return json.dumps({"error": "错误：除数不能为零"}, ensure_ascii=False)
    else:
        return json.dumps({"error": "错误：无效的运算符"}, ensure_ascii=False)

# 为每个工具定义pydantic参数校验模型
# 作用：定义参数的标准答案，字段名，类型，描述，必填项，取值范围
# 模型输出的参数必须完全符合这个定义才算合法
class GetWeatherArgs(BaseModel):
    city: str = Field(description="要查询的城市名称，例如：北京、上海")

class CalculatorArgs(BaseModel):
    a: float = Field(description="第一个运算数字")
    b: float = Field(description="第二个运算数字")
    operator: str = Field(description="运算符，必须是+、-、*、/中的一个")

# 工具描述, 直接从pydantic模型生成schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            # 直接从pydantic模型生成schema，无需手动定义
            "parameters": GetWeatherArgs.model_json_schema()
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "简易的计算器，计算两个数的和、差、积、商",
            "parameters": CalculatorArgs.model_json_schema()
        }
    }
]

# 工具执行映射表：新增绑定对应的参数校验模型
tool_map = {
    "get_weather":{
        "function": get_weather,
        "param_model": GetWeatherArgs
    },
    "calculator":{
        "function": calculator,
        "param_model": CalculatorArgs
    }
}

# 参数校验函数
def validate_tool_args(tool_name: str, tool_args: dict) -> tuple[bool, dict|str]:
    """
    校验工具参数是否合法
    :param tool_name:工具名称
    :param tool_args:模型输出的参数字典
    :return:元组(是否合法, 合法返回参数校验后的信息/不合法返回错误信息)
    """
    if tool_name not in tool_map:
        return False, json.dumps({"error": f"错误：工具 {tool_name} 不存在"}, ensure_ascii=False)
    # 获取参数校验模型
    param_model = tool_map[tool_name]["param_model"]
    try:
        # 核心：用pydantic模型校验参数，自动检查类型、必填字段等
        valid_args = param_model.model_validate(tool_args)
        # 校验通过，返回转成字典的合法参数
        return True, valid_args.model_dump()
    except ValidationError as e:
        # 校验失败，把具体错误拼接成可读的提示，返回给模型纠错
        error_msgs = []
        for err in e.errors():
            field = ",".join(err["loc"])
            msg = err["msg"]
            error_msgs.append(f"字段 {field} 错误：{msg}")
        error_text = "\n".join(error_msgs)
        return False, json.dumps({"error": f"校验参数失败：\n{error_text}\n请修正参数后重新调用工具"}, ensure_ascii=False)

# 工具执行函数
# 核心改变，给执行工具函数增加重试逻辑，
def execute_tool(tool_name: str, tool_args: dict) -> str:
    """带重试的工具执行函数，执行失败自动重试"""
    if tool_name not in tool_map:
        return json.dumps({"error": f"错误：工具 {tool_name} 不存在"}, ensure_ascii=False)

    for attempt in range(MAX_RETRY):
        try:
            result = tool_map[tool_name]["function"](**tool_args)
            return json.dumps({"result": result}, ensure_ascii=False)
        except Exception as e:
            # 如果工具执行失败，进行重试
            if attempt < MAX_RETRY - 1:
                # 重试前等待一小段时间（指数退避可以在这里加入）
                time.sleep(0.5*(attempt+1))
                continue
            # 重试全部失败，返回错误给模型
            return json.dumps({"error": f"错误：工具 {tool_name} 调用失败，已重试{MAX_RETRY}次，参数 {tool_args}，最后一次错误信息：{str(e)}"}, ensure_ascii=False)

system_prompt = """你是一个智能助手，可以调用工具获取信息来准确回答用户的问题。
请先思考是否需要调用工具，信息足够时直接回答用户。"""

class Agent:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    # 加载历史对话记录
    def _load_history(self):
        file_path = get_abs_path("history_messages/history_messages.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    messages = json.load(f)
                # 确保历史会话第一条是系统提示
                if messages and messages[0]["role"] == "system":
                    return messages
            except Exception as e:
                return json.dumps({"error": f"错误：加载历史记录失败，错误信息 {str(e)}"})
        return [{"role": "system", "content": system_prompt}]

    # 保存会话记录
    def _save_history(self, messages: list[dict]):
        """保存会话历史"""
        file_path = get_abs_path("history_messages/history_messages.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": f"错误：保存历史记录失败，错误信息 {str(e)}"})

    # 运行核心逻辑
    def run(self, query: str, max_cycle: int = 8):
        # 加载历史会话记录，同时添加当前用户问题
        messages = self._load_history()
        messages.append({"role": "user", "content": query})

        # 核心循环，带有参数校验和自纠错
        for i in range(max_cycle):
            print(f"\n{'=' * 30} 第 {i + 1} 轮 {'=' * 30}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
                stream=False
            )

            assistant_msg = response.choices[0].message
            print(f"模型回复内容：{assistant_msg.content}")
            messages.append(assistant_msg.model_dump())

            # 若没有调用工具，则直接返回结果
            if not assistant_msg.tool_calls:
                print("✅ 模型给出最终答案，结束循环")
                self._save_history(messages)
                return assistant_msg.content

            # 调用工具，将调用的工具循环，得到所用工具的参数和返回结果
            all_valid = True
            for tool_call in assistant_msg.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                # 如果JSON解析失败，直接反馈错误让模型修正
                except json.JSONDecodeError:
                    all_valid = False
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps({"error": "错误：工具参数JSON解析失败，请修正参数后重新调用工具"})
                        }
                    )
                    continue
                print(f"🔧 调用工具：{tool_name}，原始参数：{tool_args}")
                # 核心新增，参数校验
                is_valid, result = validate_tool_args(tool_name, tool_args)
                # 如果参数校验失败，将结果反馈给模型让模型修正
                if not is_valid:
                    all_valid = False
                    print(f"❌ 参数校验失败：{result}")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        }
                    )
                # 如果参数校验成功，调用工具并记录结果
                else:
                    print(f"✅ 参数校验通过，执行工具")
                    tool_result = execute_tool(tool_name, tool_args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        }
                    )
            self._save_history(messages)
            # 如果所有工具都校验通过且执行完成，继续下一轮思考；如果有错误，下一轮模型会自动纠错
            if all_valid:
                print("✅ 本轮所有工具调用均合法执行")

        # 超出最大轮次
        return json.dumps({"error": "抱歉，多次尝试后仍未得出结果，请换个方式提问。"})

if __name__ == '__main__':
    from utils.config_tool import agent_conf
    api_key = os.getenv("API_KEY_qwen")
    agent = Agent(api_key=api_key, base_url=agent_conf["base_url"], model=agent_conf["model"])
    question = "再查一下南京的天气，帮我算一下 100 除以 0"
    res = agent.run(question)
    print(f"\n{'=' * 30} 最终回答 {'=' * 30}")
    print(res)