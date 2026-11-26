"""
会话管理器：实现多会话隔离，每个会话拥有独立的agent和记忆
设计原则：无状态组件全局共享，有状态组件会话隔离
"""
import os
import uuid
import time
import json
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
    agent: Optional[PlanAgent]    # 每个会话独立的 FSM Agent 实例 ★ 改为可选，懒加载
    create_time: float = field(default_factory=time.time)
    last_active_time: float = field(default_factory=time.time)

    def touch(self) -> None:
        """更新最后活跃时间，每次访问会话都调用一次"""
        self.last_active_time = time.time()

    def to_meta(self) -> dict:
        """★ 新增：只序列化元数据，绝不序列化 agent"""
        return {
            "session_id": self.session_id,
            "create_time": self.create_time,
            "last_active_time": self.last_active_time,
        }

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
            session_timeout: int = 36000,
            # 最大同时存在的会话数量，防止内存爆掉
            max_sessions: int = 100,
            # 是否开启每个会话的断点快照
            enable_checkpoint: bool = True,
            # ========== 新增：记忆策略配置 ==========
            memory_type: str = "sliding_window",
            memory_policy_config: MemoryPolicyConfig = None,
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

        # ★ 新增：registry 持久化路径
        self._registry_path = get_abs_path("history_messages/sessions/_registry.json")
        # ★ 新增：启动时从磁盘加载元数据（不重建 agent，毫秒级）
        self._load_registry()

        # 新增：保存记忆配置
        self.memory_type = memory_type
        self.memory_policy_config = memory_policy_config

        logger.info(
            f"会话管理器初始化完成 | "
            f"超时={session_timeout}s | 最大会话数={max_sessions} | "
            f"断点快照={'开启' if enable_checkpoint else '关闭'} | "
            f"记忆模式={memory_type}"
        )

    # ===================== ★ 新增：registry 持久化 =====================

    def _load_registry(self) -> None:
        """启动时加载元数据，不创建任何 agent（毫秒级）"""
        if not os.path.exists(self._registry_path):
            # 首次启动或旧数据升级：扫目录生成初始 registry
            self._migrate_legacy_sessions()
            return

        try:
            with open(self._registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sid, meta in data.items():
                self._sessions[sid] = SessionInfo(
                    session_id=sid,
                    create_time=meta.get("create_time", time.time()),
                    last_active_time=meta.get("last_active_time", time.time()),
                    agent=None,   # ★ 懒加载：不重建 agent
                )
            logger.info(f"会话注册表加载完成，共 {len(self._sessions)} 个历史会话（懒加载模式）")
        except Exception as e:
            logger.error(f"加载会话注册表失败：{e}，将以空注册表启动")
            self._sessions = {}

    def _save_registry(self) -> None:
        """原子写入 registry，只写元数据"""
        try:
            os.makedirs(os.path.dirname(self._registry_path), exist_ok=True)
            data = {sid: info.to_meta() for sid, info in self._sessions.items()}
            tmp_path = self._registry_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._registry_path)
        except Exception as e:
            # 持久化失败不影响内存使用，只记 warning
            logger.warning(f"保存会话注册表失败：{e}")

    def _migrate_legacy_sessions(self) -> None:
        """首次启动兼容旧数据：扫描磁盘生成初始 registry"""
        sessions_root = get_abs_path("history_messages/sessions")
        if not os.path.isdir(sessions_root):
            return

        migrated = 0
        for entry in os.listdir(sessions_root):
            session_dir = os.path.join(sessions_root, entry)
            # 跳过非目录和 registry 文件
            if not os.path.isdir(session_dir) or entry.startswith("_"):
                continue
            # 只有包含快照的目录才算有效会话
            if not os.path.exists(os.path.join(session_dir, "agent_snapshot.json")):
                continue

            self._sessions[entry] = SessionInfo(
                session_id=entry,
                create_time=os.path.getctime(session_dir),
                last_active_time=os.path.getmtime(session_dir),
                agent=None,
            )
            migrated += 1

        if migrated > 0:
            self._save_registry()
            logger.info(f"首次迁移：扫描生成注册表，共 {migrated} 个历史会话")

    def _create_session_memory(self, session_id: str):
        """
        为单个会话创建独立的记忆实例
        每个会话的记忆都是全新的，完全隔离
        根据 memory_type 自动选择滑动窗口 / 摘要记忆
        """
        # 共用的策略配置，两种记忆复用同一套分级规则
        policy_config = self.memory_policy_config

        if self.memory_type == "summary":
            # ========== 摘要记忆模式 ==========
            from utils.config_tool import agent_conf
            base_memory = SummaryMemory(
                llm_client=self.llm_client,
                chat_model=agent_conf["chat_model"],
                config=MemoryConfig(
                    keep_recent_rounds=6,  # 保留最近6轮原文
                    summary_token_threshold=8000,  # 超过8k token触发摘要
                    max_summary_tokens=1000,  # 单条摘要最大token
                    enable_incremental_summary=True,  # 开启增量摘要
                ),
                policy_config=policy_config,
            )
        else:
            # ========== 默认：滑动窗口模式 ==========
            base_memory = SlidingWindowMemory(
                policy_config=policy_config
            )

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

        # ★ 新增：把新会话元数据落盘
        self._save_registry()

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
            self._save_registry()  # ★ 新增：过期删除同步到磁盘
            return None

        # 更新最后活跃时间
        session_info.touch()
        return session_info

    def get_agent(self, session_id: str) -> Optional[PlanAgent]:
        """
        快捷方法：根据会话id获取对应的agent实例
        获取 agent 实例（懒加载）
        首次访问时从磁盘重建 agent，后续直接命中缓存
        """
        session_info = self.get_session(session_id)
        if session_info is None:
            return None

        # ★ 新增：懒加载重建 agent
        if session_info.agent is None:
            try:
                logger.info(f"懒加载重建 agent：{session_id}")
                session_info.agent = self._build_agent(session_id)
                # 重建完成后立即刷新活跃时间
                session_info.touch()
            except Exception as e:
                logger.error(f"重建 agent 失败 {session_id}：{e}")
                return None

        return session_info.agent

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
        """手动删除指定会话，同时清理快照文件"""
        if session_id not in self._sessions:
            return False

        session_info = self._sessions[session_id]
        agent = session_info.agent

        # ★ 兼容 agent=None（懒加载还没触发）：用约定路径找快照
        if agent is not None and hasattr(agent, "snapshot_path"):
            snapshot_path = agent.snapshot_path
        else:
            snapshot_path = get_abs_path(
                f"history_messages/sessions/{session_id}/agent_snapshot.json"
            )

        if os.path.exists(snapshot_path):
            try:
                os.remove(snapshot_path)
                logger.debug(f"删除会话快照：{snapshot_path}")
            except Exception as e:
                logger.warning(f"清理会话快照失败：{e}")

            self._sessions.pop(session_id)
            logger.info(f"手动删除会话：{session_id}")
            return True
        return False