"""
会话管理器：实现多会话隔离，每个会话拥有独立的agent和记忆
设计原则：无状态组件全局共享，有状态组件会话隔离
"""
import os
import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

# 导入你已有的模块
from plan_and_execute.planner import Planner
from plan_and_execute.executor import Executor
from plan_and_execute.reflector import Reflector
from agent_code.agent_with_plan_and_fsm import PlanAgent
from client.llm_client import LLMClient
from tools.tool_retriever import ToolRetriever
from memory.sliding_window import SlidingWindowMemory
from memory.long_term_memory import LongTermMemory
from utils.config_tool import agent_conf
from utils.logger_tool import logger
from utils.path_tool import get_abs_path
# 新增摘要记忆相关类
from memory.summary_memory import SummaryMemory
from pydantic_schemas.memory_schemas import MemoryConfig
from pydantic_schemas.memory_policy_schema import MemoryPolicyConfig


@dataclass
class SessionInfo:
    """
    会话信息结构体：保存单个会话的所有状态
    所有和会话相关的数据都存在这里，统一管理
    """
    session_id: str
    agent: PlanAgent    # 每个会话独立的 FSM Agent 实例
    create_time: float = field(default_factory=time.time)
    last_active_time: float = field(default_factory=time.time)

    def touch(self) -> None:
        """更新最后活跃时间，每次访问会话都调用一次"""
        self.last_active_time = time.time()

class SessionManager:
    """
    会话管理器：全局单例，负责所有会话的创建、查询、清理
    外部只和这个类交互，不用关心内部怎么创建agent
    """
    def __init__(
            self,
            llm_client: LLMClient,
            tool_retriever: ToolRetriever,
            planner: Planner,
            executor: Executor,
            reflector: Reflector,
            # 会话过期时间，单位秒，默认三十分钟无活动就过期
            session_timeout: int = 1800,
            # 最大同时存在的会话数量，防止内存爆掉
            max_sessions: int = 100,
            # 是否开启每个会话的断点快照
            enable_checkpoint: bool = True,
    ):
        """
        初始化：接受所有全局共享的无状态组件
        这些组件只在程序启动时创建一次，所有会话共用
        """
        # 全局共享的无状态组件
        self.llm_client = llm_client
        self.tool_retriever = tool_retriever
        self.planner = planner
        self.executor = executor
        self.reflector = reflector

        # 会话配置
        self.session_timeout = session_timeout
        self.max_sessions = max_sessions
        self.enable_checkpoint = enable_checkpoint

        # 会话注册表，核心存储，key = session_id，value = SessionInfo
        self._sessions: Dict[str, SessionInfo] = {}

        logger.info(
            f"会话管理器初始化完成 | "
            f"超时={session_timeout}s | 最大会话数={max_sessions} | "
            f"断点快照={'开启' if enable_checkpoint else '关闭'}"
        )

    def _create_session_memory(self, session_id: str):
        """
        为单个会话创建独立的记忆实例
        每个会话的记忆都是全新的，完全隔离
        """
        # 短期记忆，每个会话独立一份滑动窗口记忆
        base_memory = SlidingWindowMemory()

        # 长期记忆：MVP项目阶段，可以先按照会话隔离，每个会话有独立的向量库目录
        # 后面做全局知识库时，可以改成共享全局向量库
        long_term_memory = LongTermMemory(persist_dir=f"chroma_db/sessions/{session_id}")

        return base_memory, long_term_memory

    def _build_agent(self, session_id) -> PlanAgent:
        """
        为指定会话创建独立的 FSM PlanAgent实例
        复用全局组件，只创建会话独有的记忆和快照路径
        """
        base_memory, long_term_memory = self._create_session_memory(session_id)

        # ★ 新增 1：给这个会话建专属目录
        session_dir = get_abs_path(f"history_messages/sessions/{session_id}")
        os.makedirs(session_dir, exist_ok=True)

        # ★ 新增 2：快照也放进这个专属目录（原来放在 snapshots/ 下）
        snapshot_path = os.path.join(session_dir, "agent_snapshot.json")

        agent = PlanAgent(
            llm_client=self.llm_client,
            retriever=self.tool_retriever,
            planner=self.planner,
            executor=self.executor,
            reflector=self.reflector,
            base_memory=base_memory,
            long_term_memory=long_term_memory,
            # 传入独立快照路径
            snapshot_path=snapshot_path,
            # 全局配置是否开启断点
            enable_checkpoint=self.enable_checkpoint,
            storage_dir=session_dir,  # ★ 新增 3：把专属目录传给 PlanAgent
        )
        return agent

    def create_session(self) -> str:
        """
        创建一个新会话，返回session_id
        创建前先清理过期会话，控制总数量
        """
        # 先清理一波过期会话，腾位置
        self.clean_expired()

        # 防御：如果会话数还是超限，就报错
        if len(self._sessions) >= self.max_sessions:
            raise RuntimeError(f"会话数量已达上限 {self.max_sessions}，请稍后再试")

        # 生成全局唯一的会话ID
        session_id = str(uuid.uuid4())

        # 创建该会话专属的Agent实例
        agent = self._build_agent(session_id)

        # 封装成SessionInfo并存入注册表
        session_info = SessionInfo(session_id, agent)
        self._sessions[session_id] = session_info

        logger.info(f"创建新会话：{session_id}，当前总会话数：{len(self._sessions)}")
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """
        根据session_id获取会话信息
        存在则更新活跃时间并返回会话信息，不存在则返回None
        """
        session_info = self._sessions.get(session_id)
        if not session_info:
            logger.warning(f"会话不存在：{session_id}")
            return None

        # 检查是否过期
        if time.time() - session_info.last_active_time > self.session_timeout:
            logger.info(f"会话已过期，自动移除：{session_id}")
            self._sessions.pop(session_id, None)
            return None

        # 更新最后活跃时间
        session_info.touch()
        return session_info

    def get_agent(self, session_id: str) -> Optional[PlanAgent]:
        """快捷方法：根据会话id获取对应的agent实例"""
        session_info = self.get_session(session_id)
        return session_info.agent if session_info else None

    def clean_expired(self) -> int:
        """
        清理所有过期会话
        返回清理掉的会话数量
        """
        now = time.time()
        expired_ids = [
            sid for sid, session_info in self._sessions.items()
            if now - session_info.last_active_time > self.session_timeout
        ]

        for sid in expired_ids:
            self._sessions.pop(sid, None)
            logger.debug(f"清理过期会话：{sid}")

        if expired_ids:
            logger.info(f"清理过期会话完成，共清理 {len(expired_ids)} 个，剩余 {len(self._sessions)} 个")
        return len(expired_ids)

    def remove_session(self, session_id: str) -> bool:
        """
        手动删除指定会话
        同时清理该会话的快照文件，避免残留
        """
        if session_id in self._sessions:
            # 额外清理快照文件
            agent = self._sessions[session_id].agent
            if hasattr(agent, "snapshot_path"):
                import os
                if os.path.exists(agent.snapshot_path):
                    try:
                        os.remove(agent.snapshot_path)
                        logger.debug(f"删除会话快照：{agent.snapshot_path}")
                    except Exception as e:
                        logger.warning(f"清理会话快照失败：{e}")

            self._sessions.pop(session_id)
            logger.info(f"手动删除会话：{session_id}")
            return True
        return False