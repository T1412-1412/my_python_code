"""
API 路由层：所有接口定义在这里
职责：接受HTTP请求 -> 参数校验 -> 调用业务层 -> 封装响应返回
原则：不写业务逻辑，只做协议转换和编排
"""

from fastapi import APIRouter, Depends, HTTPException
from api.schemas import(
    ApiResponse,
    SessionCreateResponse,
    ChatResponse,
    ChatRequest,
    HistoryMessage,
    HistoryResponse,
)
from api.depends import get_session_manager
from session.session_manager import SessionManager
from utils.logger_tool import logger


# 创建路由前缀，统一管理版本
router = APIRouter(prefix="/api/v1", tags=["会话管理"])

# 1、创建会话
@router.post("/sessions", response_model=ApiResponse)
def create_session(session_manager: SessionManager = Depends(get_session_manager)):
    """
    创建一个新的对话
    返回session_id，后续所有请求都需要携带此ID
    """
    try:
        session_id = session_manager.create_session()
        return ApiResponse(
            code=0,
            message="会话创建成功",
            data=SessionCreateResponse(session_id=session_id),
        )
    except Exception as e:
        logger.error(f"创建会话失败：{e}")
        raise HTTPException(status_code=500, detail=f"创建会话失败：{str(e)}")

# 2、发送聊天信息
@router.post("/sessions/{session_id}/chat", response_model=ApiResponse)
def chat(
    session_id: str,
    request: ChatRequest,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """
    向指定会话发送消息，获取agent回答
    - session_id: 路径参数会话唯一ID
    - request: 请求体，包含用户提问
    """
    # 第一步：校验会话是否存在
    agent = session_manager.get_agent(session_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在或已过期")

    try:
        logger.info(f"收到会话[{session_id}]消息：{request.query[:50]}...")

        # 第二步：调用 Agent 核心逻辑，获取回答
        # 注意：这里用普通def，FastAPI会自动放到线程池执行，不会阻塞事件循环
        answer = agent.run(query=request.query, resume=request.resume)

        # 第三步：封装返回结果
        return ApiResponse(
            code=0,
            message="success",
            data=ChatResponse(
                answer=answer,
                session_id=session_id
            )
        )
    except Exception as e:
        logger.error(f"会话[{session_id}]执行失败：{e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"执行失败：{str(e)}")

# 3. 获取会话历史
@router.get("sessions/{session_id}/history", response_model=ApiResponse)
def get_history(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """获取指定会话的完整历史消息"""
    session_info = session_manager.get_session(session_id)
    if session_info is None:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在或已过期")
    try:
        # 从记忆中获取历史消息
        # 注意！这里泄露了业务层面的信息，路由里面不应该知道具体业务逻辑，应该额外给出获取历史消息的接口
        history = session_info.agent.base_memory.history
        # 转换为标准格式，过滤掉元数据
        messages = [
            HistoryMessage(
                role=msg.get("role", "unknown"),
                content=msg.get("content", ""),
            )
            for msg in history
            if msg.get("role") in ("user", "assistant") # 只返回用户和助手消息，工具消息可选
        ]

        return ApiResponse(
            code=0,
            message="success",
            data=HistoryResponse(
                session_id=session_id,
                messages=messages,
                total=len(history),
            )
        )
    except Exception as e:
        logger.error(f"获取历史失败：{e}")
        raise HTTPException(status_code=500, detail=f"获取历史失败：{str(e)}")

# 4、删除会话
@router.delete("sessions/{session_id}", response_model=ApiResponse)
def delete_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """手动删除指定会话，释放资源"""
    success = session_manager.remove_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")

    return ApiResponse(
        code=0,
        message="success",
        data=None
    )