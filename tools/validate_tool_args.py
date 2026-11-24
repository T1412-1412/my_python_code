import json
from pydantic import ValidationError
from tools.tool_registry import tool_registry
from utils.logger_tool import logger


# 参数校验函数
def validate_tool_args(tool_name: str, tool_args: dict) -> tuple[bool, dict|str]:
    """
    校验工具参数是否合法
    :param tool_name:工具名称
    :param tool_args:模型输出的参数字典
    :return:元组(是否合法, 合法返回参数校验后的信息/不合法返回错误信息)
    """
    if tool_name not in tool_registry:
        logger.error(f"错误：工具 {tool_name} 不存在")
        return False, f"错误：工具 {tool_name} 不存在"
    # 获取参数校验模型
    param_model = tool_registry[tool_name]["param_model"]
    try:
        # 核心：用pydantic模型校验参数，自动检查类型、必填字段等
        valid_args = param_model.model_validate(tool_args)
        # 校验通过，返回转成字典的合法参数
        logger.info(f"工具 {tool_name} 参数校验通过：{valid_args.model_dump()}")
        return True, valid_args.model_dump()
    except ValidationError as e:
        # 校验失败，把具体错误拼接成可读的提示，返回给模型纠错
        error_msgs = []
        for err in e.errors():
            field = ",".join(err["loc"])
            msg = err["msg"]
            error_msgs.append(f"字段 {field} 错误：{msg}")
        error_text = "\n".join(error_msgs)
        logger.error(f"工具 {tool_name} 参数校验失败：\n{error_text}")
        return False, f"校验参数失败：\n{error_text}\n请修正参数后重新调用工具"