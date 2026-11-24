from pydantic_schemas.tool_params_model import (
    GetWeatherArgs,
    CalculatorArgs,
    SearchKnowledgeBaseArgs,
    WriteFileArgs
)
from tools.tool_functions.calculator import calculator
from tools.tool_functions.get_weather import get_weather
from tools.tool_functions.search_knowledge_base import search_knowledge_base
# 顶部导入工具函数
from tools.tool_functions.file_operations import write_file


# 【核心】工具完整注册表，包含函数、参数模型、检索用的描述信息
tool_registry = {
    "get_weather": {
        "function": get_weather,
        "param_model": GetWeatherArgs,
        # 检索用的描述：描述越详细，检索召回越准确
        "search_text": "天气查询工具，支持查询指定城市的天气详细信息"
    },
    "calculator": {
        "function": calculator,
        "param_model": CalculatorArgs,
        "search_text": "进行简单的数学计算，支持加、减、乘、除四种运算"
    },
    # ========== 新增：知识库检索工具 ==========
    "search_knowledge_base": {
        "function": search_knowledge_base,
        "param_model": SearchKnowledgeBaseArgs,
        "search_text": "从指定知识库中检索相关文档内容，用于回答专业问题、文档内容查询、资料查找"
    },
    # ========== 新增：文件操作工具 ==========
    "write_file": {
            "function": write_file,
            "param_model": WriteFileArgs,
            "search_text": "将内容写入到本地文件中，支持覆盖写入和追加写入，用于保存结果、生成报告、导出数据"
        },
}