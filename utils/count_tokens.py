from typing import List, Dict

import tiktoken

from utils.config_tool import agent_conf


def count_tokens(messages: List[Dict]):
    """
    精确计算一组消息的总Token数
    工业级细节：不仅计算content，还要计算tool_calls的参数内容
    :param messages: 消息列表
    :return: 总Token数
    """
    # 初始化分词器：使用官方tiktoken，保证和模型侧计数一致
    try:
        encoder = tiktoken.encoding_for_model(model_name=agent_conf["chat_model"])
    except KeyError:
        # 非OpenAI模型（qwen3.7-plus）使用与模型对应的分词器编码（o200k_base）
        encoder = tiktoken.get_encoding(agent_conf["tiktoken_encoder"])

    total_tokens = 0
    for msg in messages:
        # 计算消息正文内容的token
        content = msg.get("content", "")
        if content:
            total_tokens += len(encoder.encode(content))

        # 关键：计算工具调用请求的参数Token（容易被忽略的大头）
        if "tool_calls" in msg and msg["tool_calls"]:
            for tool_call in msg["tool_calls"]:
                if "function" in tool_call:
                    # 工具名 + 参数内容 都要算Token
                    function_name = tool_call["function"].get("name", "")
                    function_args = tool_call["function"].get("arguments", "")
                    total_tokens += len(encoder.encode(function_name))
                    total_tokens += len(encoder.encode(function_args))

        # 工具回复消息的ID也会占用少量Token，精确场景下可补充计算
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            total_tokens += len(encoder.encode(msg["tool_call_id"]))

    return total_tokens