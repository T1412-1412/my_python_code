"""
API 数据模型：定义所有接口的请求和响应格式
核心思想：契约先行，所有的输入输出都有明确结构，自动校验
"""


from typing import List, Optional, Any
from pydantic import BaseModel, Field


# 基础统一响应模型
class ApiResponse(BaseModel):
    """统一响应格式：所有接口都返回这个结构，前端统一处理"""
    code: int = Field(default=0, description="状态码，0表示成功，非0表示错误")
    message: str = Field(default="success", description="提示信息")
    data: Optional[Any] = Field(default=None, description="业务数据")

# 会话相关模型
class SessionCreateResponse(BaseModel):
    """创建会话的返回数据"""
    session_id: str = Field(description="会话唯一ID，后续请求都要带上")

class ChatRequest(BaseModel):
    """发送消息的请求体"""
    query: str = Field(
        min_length=1,
        max_length=2000,
        description="用户提问内容，长度限制防止超长文本"
    )
    resume: bool = Field(
        default=True,
        description="是否尝试从断点恢复，默认开启"
    )

class ChatResponse(BaseModel):
    """聊天接口返回数据"""
    answer: str = Field(description="Agent的回答内容")
    session_id: str = Field(description="所属会话ID")

class HistoryMessage(BaseModel):
    """单条历史消息结构"""
    role: str = Field(description="消息角色：user/assistant/tool")
    content: str = Field(description="消息内容")

class HistoryResponse(BaseModel):
    """会话历史返回数据"""
    session_id: str = Field(description="会话ID")
    messages: List[HistoryMessage] = Field(description="历史消息列表")
    total: int = Field(description="消息总条数")

# ===================== 知识库相关模型 =====================
class KnowledgeBaseInfo(BaseModel):
    """知识库基本信息"""
    name: str = Field(description="知识库名称（唯一标识）")
    doc_count: int = Field(description="知识库中文档分片总数")


class KnowledgeBaseListResponse(BaseModel):
    """知识库列表返回"""
    total: int = Field(description="知识库总数")
    items: List[KnowledgeBaseInfo] = Field(description="知识库列表")


class SearchKbRequest(BaseModel):
    """知识库检索请求体"""
    query: str = Field(min_length=1, max_length=500, description="检索查询词")
    top_k: int = Field(default=3, ge=1, le=20, description="返回结果数量")


class SearchResultItem(BaseModel):
    """单条检索结果"""
    content: str = Field(description="相关文本内容")
    source: str = Field(description="来源文件名")
    score: float = Field(description="相似度分数，0~1，越大越相似")


class SearchKbResponse(BaseModel):
    """知识库检索返回"""
    kb_name: str = Field(description="知识库名称")
    results: List[SearchResultItem] = Field(description="检索结果列表")


class UploadDocumentResponse(BaseModel):
    """文档上传返回"""
    kb_name: str = Field(description="知识库名称")
    filename: str = Field(description="上传的文件名")
    chunk_count: int = Field(description="成功入库的分片数量")
