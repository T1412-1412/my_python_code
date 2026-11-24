"""
带有会话记忆的agent，将会话历史保存到文件中，同时使用了Function Calling的格式化输出，
可以大幅减少提示词内容，大模型输出的内容格式也更稳定
"""
import os
import json
from openai import OpenAI
import requests


def get_weather(city: str):
    """
    输入城市名称，获取天气信息
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

tool_map = {
    "get_weather": get_weather,
    "calculator": calculator
}

def execute_tool(tool_name: str, tool_args_str: str):
    try:
        tool_args = json.loads(tool_args_str)
    except json.JSONDecodeError:
        return json.dumps({"error": "工具参数解析出错，不是合法JSON"}, ensure_ascii=False)
    if tool_name not in tool_map:
        return json.dumps({"error": f"没有找到该工具：{tool_name}"}, ensure_ascii=False)
    # 调用工具，参数解包传入
    return tool_map[tool_name](**tool_args)


tools = [
    {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市天气信息的工具函数，输入城市名称，返回天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city":{"type": "string", "description": "城市名称，例如：北京，上海。"}
            },
            "required": ["city"],
            "additionalProperties": False
        },
        "strict": True
    }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "一个简单的计算器，可以进行加、减、乘、除运算",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "数字1"},
                    "b": {"type": "number", "description": "数字2"},
                    "operator": {"type": "string", "description": "运算符，必须为其中之一：+、-、*、/"}
                },
                "required": ["a", "b", "operator"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

system_prompt = """你是一个智能助手，可以调用工具获取信息来准确回答用户的问题。
请先思考是否需要调用工具，信息足够时直接回答用户。"""

class Agent:
    def __init__(self, api_key: str, base_url: str, model: str, tools: list[dict], max_step: int = 5):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.tools = tools
        self.max_step = max_step
    def _load_history(self):
        """加载会话历史"""
        if os.path.exists("../history_messages/history_messages.json"):
            try:
                with open("../history_messages/history_messages.json", "r", encoding="utf-8") as f:
                    messages = json.load(f)
                # 确保第一条是系统提示词
                if messages and messages[0]["role"] == "system":
                    return messages
            except Exception as e:
                print(f"加载会话历史出错，已创建新会话列表，错误信息：{str(e)}")

        return [{"role": "system", "content": system_prompt}]

    def _save_history(self, messages: list[dict]):
        """保存会话历史"""
        with open("../history_messages/history_messages.json", "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=4)

    def run(self, query: str):
        # 加载会话历史
        messages = self._load_history()
        messages.append({"role": "user", "content": query})

        for step in range(self.max_step):
            res = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )
            llm_result = res.choices[0].message

            # 把assistant消息加入上下文
            # ========== 核心修复1：把模型对象转成标准字典，再加入上下文 ==========
            # model_dump() 是Pydantic对象自带方法，自动把类实例转为原生字典
            llm_result_dict = llm_result.model_dump()
            messages.append(llm_result_dict)

            # 没有调用工具，直接返回结果
            if not llm_result.tool_calls:
                self._save_history(messages)
                return llm_result.content
            # 调用工具
            for tool_call in llm_result.tool_calls:
                tool_name = tool_call.function.name
                tool_id = tool_call.id
                # 必须解析，否则传入的参数为字符串，或者在执行函数中解析
                tool_args = tool_call.function.arguments
                # 执行工具，获得工具的输出结果，加入消息列表
                tool_result = execute_tool(tool_name, tool_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": tool_result
                })
            # 每轮工具执行结束后，保存会话历史
            self._save_history(messages)
        # 超过最大思考步数，返回的结果也要保存的会话历史中
        final_answer = "达到最大思考步数，未获得结果"
        messages.append({"role": "assistant", "content": final_answer})
        self._save_history(messages)
        return final_answer

if __name__ == '__main__':
    api_key = os.getenv("API_KEY_qwen")
    base_url = "https://ws-75ntouay5nt8xgfs.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    model = "qwen3.7-max"
    agent = Agent(api_key, base_url, model, tools)
    result = agent.run("我有几只宠物，你还记得吗，计算一下100除以0等于多少")
    print(result)

