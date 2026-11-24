"""
依赖注入模块：管理全局单例组件
核心思想：无状态组件全局一份，通过依赖注入传给路由，不用每个接口都初始化
"""

from rag_knowledge.kb_singleton import get_global_kb_manager

from typing import Generator

# 导入你已有的所有核心组件
from client.llm_client import LLMClient
from tools.tool_retriever import ToolRetriever
from tools.tool_registry import tool_registry
from plan_and_execute.planner import Planner
from plan_and_execute.executor import Executor
from plan_and_execute.reflector import Reflector
from session.session_manager import SessionManager
from utils.logger_tool import logger


# 全局单例组件（只初始化一次）
# 注意，这些都是无状态组件，程序启动时创建一次，全程复用
_llm_client = None
_tool_retriever = None
_planner = None
_executor = None
_reflector = None
_session_manager = None
_initialized = False


def _init_global_components() -> None:
    """
    初始化所有全局组件
    程序启动时调用一次，懒加载模式，第一次访问才初始化
    """
    global _llm_client, _tool_retriever, _planner, _executor, _reflector, _session_manager, _initialized

    if _initialized:
        return  # 已经初始化过，直接返回

    logger.info("正在初始化全局核心组件...")
    # 1. LLM 客户端
    _llm_client = LLMClient()

    # 2. 工具检索器
    _tool_retriever = ToolRetriever(tool_registry=tool_registry)

    # 3. 规划器
    _planner = Planner(llm_client=_llm_client)

    # 4. 执行器
    _executor = Executor(llm_client=_llm_client, tool_retriever=_tool_retriever)

    # 5. 反思器重规划器
    _reflector = Reflector(llm_client=_llm_client)

    # 6. 会话管理器（核心）
    _session_manager = SessionManager(
        llm_client=_llm_client,
        tool_retriever=_tool_retriever,
        planner=_planner,
        executor=_executor,
        reflector=_reflector,
        session_timeout=1800,  # 30分钟超时
        max_sessions=100,
    )
    # ★ 不再 new，直接触发单例初始化，保证和工具层用同一个
    get_global_kb_manager()

    logger.info("全局组件初始化完成")


def get_session_manager() -> Generator[SessionManager, None, None]:
    """
    FastAPI依赖函数：返回会话管理器实例
    每个请求都会调用，但内部是单例，不会重复调用
    """
    _init_global_components()
    yield _session_manager

# 新增：依赖函数
def get_kb_manager() -> Generator[KnowledgeBaseManager, None, None]:
    """
    FastAPI 依赖函数：返回知识库管理器实例
    """
    _init_global_components()
    yield _kb_manager