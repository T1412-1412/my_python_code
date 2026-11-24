"""
工具的定义和使用：
    1. 确认大模型是否支持工具调用
    2. 定义工具函数
    3. 描述工具函数，使用json格式描述工具函数，包含工具名称、功能、参数等信息
    4. 封装大模型对话函数
    5. 调用大模型进行对话，需要工具，就调用，不需要则直接输出结果
"""
import json
import requests
from openai import OpenAI
import os


# # 1. 已经写好的工具函数
# # 定义工具函数
# def get_weather(city: str):
#     """
#     天气查询函数，输入城市名称，返回天气信息
#     :param city:要查询的城市名称
#     :return:格式化后的天气信息字符串
#     """
#     url = f"https://wttr.in/{city}?format=j1"
#
#     # 1. 发送HTTP请求，获取天气JSON数据
#     response = requests.get(url)
#
#     # 2. 解析JSON数据，提取我们需要的核心信息
#     weather_data = response.json()
#
#     # 3. 对天气信息进行格式化处理，返回JSON字符串
#     return json.dumps({
#         "城市": city,
#         "天气状况": current["weatherDesc"][0]["value"],
#         "温度": f"{current["temp_C"]}℃",
#         "湿度": f"{current["humidity"]}%",
#         "风速": f"{current["windspeedKmph"]} km/h"
#     }, ensure_ascii=False)
#
# # 描述工具函数
# # 2. 给大模型写的「工具说明书」（核心！必须严格按格式写）
# # 这是一个固定格式的列表，大模型靠它知道：有什么工具、怎么传参、什么时候用
# tools = [
#     {
#         "type": "function",
#         "function": {
#             # 工具函数的名字，必须和你定义的函数名完全一致
#             "name": "get_weather",
#             # 工具的功能描述，大模型靠这个判断什么时候调用这个工具
#             "description": "查询指定城市的实时天气信息，当用户问天气相关问题时使用",
#             # 函数的参数定义，用JSON Schema格式描述
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "city": {
#                         "type": "string",
#                         "description": "要查询天气的城市名称，例如：北京、上海、广州"
#                     }
#                 },
#                 # 必填参数
#                 "required": ["city"],
#                 "additionalProperties": False
#             },
#             "strict": True
#         }
#     }
# ]
#
# # 3. 核心联动逻辑：大模型 ↔ 工具函数
# def chat_with_weather_tool(user_query: str) -> str:
#     # 初始化大模型客户端
#     client = OpenAI(api_key=os.getenv("API_KEY_qwen"), base_url="https://ws-75ntouay5nt8xgfs.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
#
#     # 第一轮对话：用户提问 + 带上工具定义
#     messages = [
#         {"role": "system", "content": "你是一个智能助手，需要查询天气时请调用工具，不要自己编造天气数据。"},
#         {"role": "user", "content": user_query}
#     ]
#
#     # 第一次调用大模型，让它判断要不要调用工具
#     response = client.chat.completions.create(
#         model="qwen3.7-max",
#         messages=messages,
#         tools=tools,  # 关键：把工具说明书传给大模型
#         tool_choice="auto"  # 让大模型自动决定是否调用工具
#     )
#
#     response_message = response.choices[0].message
#
#     # 判断：大模型是不是要求调用工具
#     if response_message.tool_calls:
#         # 1. 把大模型的回复先加入对话历史
#         messages.append(response_message)
#
#         # 2. 遍历所有要调用的工具（一般一次只调用一个）
#         for tool_call in response_message.tool_calls:
#             function_name = tool_call.function.name
#             # 解析大模型传过来的参数
#             function_args = json.loads(tool_call.function.arguments)
#
#             print(f"[调试] 大模型调用了工具：{function_name}，参数：{function_args}")
#
#             # 3. 执行对应的工具函数（核心：把大模型的指令变成真实执行）
#             if function_name == "get_weather":
#                 function_response = get_weather(**function_args)
#             else:
#                 function_response = "未知工具"
#
#             # 4. 把工具执行结果，以「tool」角色加回对话历史
#             messages.append({
#                 "role": "tool",
#                 "tool_call_id": tool_call.id,
#                 "name": function_name,
#                 "content": function_response,
#             })
#
#         # 5. 第二次调用大模型：让它根据工具返回的结果，生成最终回答
#         second_response = client.chat.completions.create(
#             model="qwen3.7-max",
#             messages=messages
#         )
#         return second_response.choices[0].message.content
#
#     # 如果不需要调用工具，直接返回大模型的回答
#     else:
#         return response_message.content

# 1. 定义工具函数
def get_weather(city: str):
    """
    天气查询函数，输入城市名称，返回天气信息
    :param city: 城市名称，字符串，例如：北京、上海、广州
    :return: 天气信息，格式化后的JSON字符串
    """
    url = f"https://wttr.in/{city}?format=j1"
    response = requests.get(url, timeout=30)
    weather_data = response.json()
    return json.dumps({
        "城市": city,
        "天气状况": weather_data["current_condition"][0]["weatherDesc"][0]["value"],
        "温度": f"{weather_data['current_condition'][0]['temp_C']}℃",
        "湿度": f"{weather_data['current_condition'][0]['humidity']}%",
        "风速": f"{weather_data['current_condition'][0]['windspeedKmph']} km/h"
    },ensure_ascii=False)

# 2. 描述工具函数
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的天气信息，输入城市名称，返回天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "要查询天气的城市名称，例如：北京，上海，广州"
                },
                "required": ["city"],
                "additionalProperties": False
            }
        },
        "strict": True
    }
}]

# 3. 将调用大模型封装为函数，在函数内部实现用户提问--大模型判断是否使用工具--使用工具--返回结果（不使用工具则直接返回结果）
messages = [{"role": "system", "content": "你是一个智能助手，需要查询天气时请调用工具，不要自己编造天气数据。"}]
def get_model_answer(query: str):
    messages.append({"role": "user", "content": query})

    client = OpenAI(
        api_key=os.getenv("API_KEY_qwen"),
        base_url="https://ws-75ntouay5nt8xgfs.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    )

    first_response = client.chat.completions.create(
        messages=messages,
        model="qwen3.7-max",
        tools=tools,
        tool_choice="auto"
    )

    message = first_response.choices[0].message
    messages.append(message)

    if message.tool_calls:  # 判断大模型是否调用工具
        tool_name = message.tool_calls[0].function.name
        tool_id = message.tool_calls[0].id
        tool_args = json.loads(message.tool_calls[0].function.arguments)
        print(f"[调试] 大模型调用了工具：{tool_name}，参数：{tool_args}")
        tool_result = get_weather(**tool_args)

        messages.append({"role": "tool", "tool_call_id": tool_id, "name": tool_name, "content": tool_result})

        second_response = client.chat.completions.create(
            messages=messages,
            model="qwen3.7-max",
            tools=tools,
            tool_choice="auto"
        )
        return second_response.choices[0].message.content
    else:   # 大模型没有调用工具，直接返回结果
        return message.content


if __name__ == '__main__':
    print(get_model_answer("郑州的天气"))
