from pydantic import BaseModel, Field

# 工具函数的Pydantic参数校验模型
class GetWeatherArgs(BaseModel):
    city: str = Field(description="要查询的城市名称，例如：北京、上海")

class CalculatorArgs(BaseModel):
    a: float = Field(description="第一个运算数字")
    b: float = Field(description="第二个运算数字")
    operator: str = Field(description="运算符，必须是+、-、*、/中的一个")

# 在文件末尾追加
class SearchKnowledgeBaseArgs(BaseModel):
    kb_name: str = Field(description="要检索的知识库名称")
    query: str = Field(description="用于检索的问题或关键词")
    top_k: int = Field(default=3, description="返回最相关的片段数量，默认3条")

# 在文件末尾追加
class WriteFileArgs(BaseModel):
    file_path: str = Field(
        description="要写入的文件相对路径，例如 'result/quote.txt'，只能是相对路径，不能以/开头"
    )
    content: str = Field(
        description="要写入的文件内容"
    )
    append: bool = Field(
        default=False,
        description="是否追加模式，False为覆盖写入，True为追加到文件末尾"
    )
