from tools.tool_registry import tool_registry
from utils.logger_tool import logger


# 生成工具的OpenAI Schema
def build_tool_schema(tool_name: list) -> list:
    """
    根据召回的工具名列表，生成对应的OpenAI Function Calling格式
    只把召回的工具传给模型，而不是全部
    """
    tool_schema = []
    for name in tool_name:
        info = tool_registry[name]
        tool_schema.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["search_text"],
                    "parameters": info["param_model"].model_json_schema()
                }
            }
        )
        # print("="*30)
        # print(type(tool_schema))
        # print("=" * 30)
    logger.info(f"生成的工具Schema：{tool_schema}")
    return tool_schema