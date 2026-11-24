from typing import List, Dict
import tiktoken
from utils.config_tool import agent_conf
from utils.count_tokens import count_tokens
from memory.validate_context_integrity import validate_context_integrity
from memory.base_memory import BaseMemory
from memory.memory_policy import MemoryPolicyMixin
from pydantic_schemas.memory_policy_schema import MemoryPolicyConfig
from utils.logger_tool import logger


class SlidingWindowMemory(BaseMemory, MemoryPolicyMixin):
    """
    滑动窗口记忆组件
    核心能力：精确控制上下文Token数量，自动截断老旧历史，保证工具调用消息完整性
    工业级特性：精确Token计数、工具消息成对校验、系统提示词永久保留
    """
    def __init__(
            self,
            system_prompt: str=agent_conf["system_prompt"],
            chat_model: str=agent_conf["chat_model"],
            max_token: int=agent_conf["max_token"],
            policy_config: MemoryPolicyConfig = None
    ):
        """
        初始化滑动窗口记忆
        :param system_prompt: 系统提示词，永久保留，不参与截断
        :param max_token: 上下文最大Token预算（含system、历史、模型输出预留）
        :param chat_model: 模型名称，用于匹配对应的分词器，保证Token计数准确
        """
        # 初始化策略，在记忆类的__init__中调用一次即可
        self._init_policy(policy_config)
        # 系统提示词：全程置顶，永不截断
        self.system_prompt = system_prompt
        # 最大Token预算：整个上下文的Token上限
        self.max_token = max_token
        # 初始化分词器：使用官方tiktoken，保证和模型侧计数一致
        try:
            self.encoder = tiktoken.encoding_for_model(model_name=chat_model)
        except KeyError:
            # 非OpenAI模型（qwen3.7-plus）使用与模型对应的分词器编码（o200k_base）
            self.encoder = tiktoken.get_encoding(agent_conf["tiktoken_encoder"])

        # 历史消息存储：只存user、assistant、tool的消息，不包含system
        self._history: List[Dict] = []

    @property
    def history(self) -> List[Dict]:
        """实现基类抽象属性，返回全量原始历史，给agent持久化使用"""
        return self._history.copy()  # 返回副本，防止外部修改内部状态


    def add_message(self, message: Dict) -> None:
        """
        添加单条消息到记忆中
        :param message: 标准消息格式字典，支持user/assistant/tool角色
        :return: None
        """
        # 消息写入前策略处理：敏感信息检测 -> 添加元数据 -> 去重，返回处理后的消息
        policy_message = self.policy_add_message(message)
        if policy_message is not None:
            self._history.append(policy_message)

    def add_messages(self, messages: List[Dict]) -> None:
        """
        批量添加多条消息到记忆中
        :param messages: 标准消息格式字典列表，支持user/assistant/tool角色
        :return: None
        """
        # 消息写入前策略处理：敏感信息检测 -> 添加元数据 -> 去重，返回处理后的消息
        policy_messages = []
        for msg in messages:
            policy_mag = self.policy_add_message(msg)
            policy_messages.append(policy_mag)
        self._history.extend(policy_messages)

    def get_context(self) -> List[Dict]:
        """
        获取滑动窗口后的完整上下文（含system + 窗口内历史）
        核心逻辑：从后往前截取历史，直到Token预算耗尽
        :return: 可直接传给大模型API的消息列表
        """
        # 1. 构造系统消息，永久置顶
        system_msg = [{"role": "system", "content": self.system_prompt}]
        # 计算系统消息暂用的token
        system_tokens = count_tokens(system_msg)

        # 2. 计算历史消息的token
        # 预留10%给模型输出，避免输出时超限，工业级做法
        available_tokens = int(self.max_token * 0.9) - system_tokens
        if available_tokens <=0:
            raise ValueError("系统提示词过长，已超出Token预算，请精简system_prompt")

        # 3. 从后往前遍历历史，优先保留最新消息
        window_history = []
        current_tokens = 0

        # 倒序遍历，最新的消息最先加入窗口
        for msg in reversed(self._history):
            # 计算当前消息的token
            msg_tokens = count_tokens([msg])

            # 判断加入当前消息后是否超出Token预算
            if current_tokens + msg_tokens > available_tokens:
                # 超出Token预算，停止加入，并打断后续循环
                break

            # 插入到列表头部（因为是倒叙遍历，保证顺序正确）
            window_history.insert(0, msg)
            current_tokens += msg_tokens

        # 4. 关键修复：保证调用消息的完整性
        window_history = validate_context_integrity(window_history)

        # 5. 拼接完整上下文并返回
        context = system_msg + window_history
        # 过滤上下文窗口：读取时策略处理：1.按优先级排序、2.分级配额淘汰、3.剥离元数据
        policy_context = self.policy_filter_context(context)
        return policy_context

    def restore_history(self, messages: List[Dict]) -> None:
        """精确恢复：不重跑 policy、不触发摘要，直接灌入 _history"""
        if not messages:
            return
        cleaned = [m for m in messages if isinstance(m, dict)]
        self._history = cleaned # 追加改为赋值，安全，也是为了状态机的恢复
        logger.info(f"恢复历史 {len(cleaned)} 条到 SlidingWindowMemory")

    def clear(self) -> None:
        """清空历史记忆，保留系统提示词"""
        self._history = []

    def get_history_length(self) -> int:
        """获取当前历史消息条数"""
        return len(self._history)



