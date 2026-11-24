import json
import time
from tools.tool_registry import tool_registry
from utils.config_tool import agent_conf
from utils.logger_tool import logger


# 工具执行函数
# 核心改变，给执行工具函数增加重试逻辑，执行失败自动重试
def execute_tool(tool_name: str, tool_args: dict) -> str|None:
    """带重试的工具执行函数，执行失败自动重试"""
    if tool_name not in tool_registry:
        logger.error(f"工具 {tool_name} 不存在")
        return f"工具 {tool_name} 不存在"

    for attempt in range(agent_conf["max_retry"]):
        try:
            logger.info(f"开始调用工具：{tool_name}，参数：{tool_args}")
            start = time.time()
            result = tool_registry[tool_name]["function"](**tool_args)
            cost = time.time() - start
            logger.info(f"工具 {tool_name} 调用完成，耗时 {cost:.2f}s，结果 {result}")
            return f"工具 {tool_name} 调用成功，结果 {str(result)}"
        except Exception as e:
            # 如果工具执行失败，进行重试
            if attempt < agent_conf["max_retry"] - 1:
                # 重试前等待一小段时间（指数退避可以在这里加入）
                logger.warning(f"工具 {tool_name} 第 {attempt} 次重试：{str(e)}")
                time.sleep(0.5*(attempt+1))
                continue
            # 重试全部失败，返回错误给模型
            logger.error(f"工具 {tool_name} 执行失败：{str(e)}")
            return f"错误：工具 {tool_name} 调用失败，已重试{agent_conf['max_retry']}次，参数 {tool_args}，最后一次错误信息：{str(e)}"