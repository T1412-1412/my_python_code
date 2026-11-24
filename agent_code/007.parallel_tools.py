"""
当多个工具调用之间没有依赖关系时（比如同时查北京的天气和上海的天气），
串行执行会浪费时间。Function Calling 原生支持一次输出多个`tool_calls`，我们可以用并行执行提升效率。
核心实现：用线程池并行执行
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from json import JSONDecodeError
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

# 并行工具调用
# 核心实现：用线程池并行执行
def run_tools_parallel(tools_call: list) -> dict:
    # 最终返回值：{tool_call_id: 执行结果字符串}
    results = {}

    # ========== 第一步：先把所有参数校验做完（串行） ==========
    # 用来存放校验通过可以执行的任务
    task_map = {}

    for tc in tools_call:
        tool_name = tc.function.name
        tool_id = tc.id
        try:
            tool_args = json.loads(tc.function.arguments)
        except JSONDecodeError:
            results[tool_id] = f"工具{tool_name}JSON解析错误"
            continue

        is_valid, valid_args = validate_tool_args(tool_name, tool_args)
        if not is_valid:
            # 参数错了，直接存错误信息，不用进入线程池
            results[tool_id] = valid_args
        else:
            task_map[tool_id] = (tool_name, valid_args)

    # ========== 第二步：把所有合法任务提交给线程池并行执行 ==========
    # 创建线程池，最多同时跑5个任务
    with ThreadPoolExecutor(max_workers=5) as executor:
        # 提货字典，把tool_id和future对应起来
        future_to_id = {}

        for tool_id, (tool_name, tool_args) in task_map.items():
            # 提交任务：告诉线程池，去执行run_tool函数，参数是tool_name和valid_args
            # submit() 立刻返回一个future对象，不用等函数跑完
            future = executor.submit(execute_tool, tool_name, tool_args)
            # 记录这个future对应的是哪个tool_id
            future_to_id[future] = tool_id

        # ========== 第三步：收集线程池执行结果 ==========
        # as_completed会迭代所有future，哪个任务先完成，就先产出哪个，它是一个迭代器，会盯着所有提交的任务
        # 哪个任务先执行完成，就先把它的 `future` 返回给你，不用按提交顺序等，效率更高
        for future in as_completed(future_to_id):
            tool_id = future_to_id[future]
            try:
                # 获取任务的执行结果
                # 如果任务执行中抛了异常，这里会重新抛出来
                tool_result = future.result()
                results[tool_id] = tool_result
            except Exception as e:
                results[tool_id] = f"工具{tool_name}执行出错：{str(e)}"
    return results

class Agent:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    # 加载历史对话记录
    def _load_history(self):
        file_path = get_abs_path("../history_messages/history_messages.json")
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
        file_path = get_abs_path("../history_messages/history_messages.json")
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

            tool_results = run_tools_parallel(assistant_msg.tool_calls)
            for tool_call in assistant_msg.tool_calls:
                tool_id = tool_call.id
                tool_result = tool_results[tool_id]
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": tool_result
                    }
                )
            self._save_history(messages)
        # 超出最大轮次
        return json.dumps({"error": "抱歉，多次尝试后仍未得出结果，请换个方式提问。"})

if __name__ == '__main__':
    from utils.config_tool import agent_conf
    api_key = os.getenv("API_KEY_qwen")
    agent = Agent(api_key=api_key, base_url=agent_conf["base_url"], model=agent_conf["model"])
    question = "查一下南京的天气，再查一下北京的天气，然后计算这两个城市的温度之和"
    res = agent.run(question)
    print(f"\n{'=' * 30} 最终回答 {'=' * 30}")
    print(res)