""" Agent主运行代码，增加滑动窗口记忆 完整修复版 """
import os
import json
import time

from client.llm_client import LLMClient
from pydantic_schemas.build_tool_schema import build_tool_schema
from tools.run_tools_parallel import run_tools_parallel
from utils.path_tool import get_abs_path
from utils.config_tool import agent_conf
from utils.logger_tool import logger
from typing import List, Dict
from memory.base_memory import BaseMemory
from memory.long_term_memory import LongTermMemory


# Agent主循环
class Agent:
    def __init__(
            self, llm_client: LLMClient,
            base_memory: BaseMemory,
            long_term_memory: LongTermMemory,
            chat_model: str, tool_retriever,
            max_cycle: int = agent_conf["max_cycle"]
        ):
        self.client = llm_client
        self.model = chat_model
        self.tool_retriever = tool_retriever
        self.max_cycle = max_cycle
        self.base_memory = base_memory
        self.long_term_memory = long_term_memory

        # 完整会话日志：只 append，永不截断，专门用于持久化
        self._session_log: List[Dict] = []
        # 幂等标志：避免重复加载磁盘历史、重复注入
        self._history_inited = False

        # 初始化时加载历史
        self._init_history_from_disk()

    def _init_history_from_disk(self) -> None:
        """
        从磁盘加载完整历史，并同步到 base_memory 和 _session_log。
        只执行一次（幂等），避免重复加载、重复注入 system。
        """
        if self._history_inited:
            return

        state_path = get_abs_path("history_messages/memory_state.json")
        # 1) 优先用快照恢复（含摘要状态）
        restored = False
        if hasattr(self.base_memory, "load_state"):
            restored = self.base_memory.load_state(state_path)

        # 2) 快照不可用，退回原有逻辑
        if not restored:
            history = self._load_history()
            if history:
                # 1) 完整历史 → _session_log，用于持久化时"全量覆盖"
                self._session_log = history
                # 2) 注入 base_memory，用于当次对话的上下文构建
                self.base_memory.restore_history(history)
                logger.info(f"历史记忆注入完成，共 {len(history)} 条")

        # 3) session_log 始终维护一份（后续持久化、审计都用得到）
        if not self._session_log and restored:
            # 用 _history 的副本重建（此时 _history 已经是最新的权威数据）
            self._session_log = self.base_memory.history

        self._history_inited = True

    def _record(self, msg: Dict) -> None:
        """
        消息统一写入入口：一条消息三路同步。
        三路 = 短期记忆(上下文) + 长期记忆(向量库) + 会话日志(持久化)

        为什么收敛成一个入口？
        1. 将来加"写入前脱敏 / 加时间戳 / 过滤敏感角色"，只改这里
        2. 保证三路永远同步，不会出现"某一路漏写"的隐蔽 bug
        3. 语义清晰：业务上"记录一条消息"就是一个动作
        """
        self.base_memory.add_message(msg)
        self.long_term_memory.add_message(msg)
        self._session_log.append(msg)

    # 加载历史对话记录：只读取磁盘，不往滑动窗口写数据
    def _load_history(self):
        file_path = get_abs_path("history_messages/history_messages.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    messages = json.load(f)
                # if isinstance(messages, list) and len(messages) > 0:
                #     # 磁盘存储第一条永远是system消息，原地pop删除，不要赋值接收pop返回值
                #     messages.pop(0)
                    logger.debug("历史记录加载成功")
                    return messages
            except Exception as e:
                logger.error(f"错误：加载历史记录失败，错误信息 {str(e)}")
                return []
        logger.debug("历史记录不存在")
        return []

    # 保存会话记录：仅会话结束调用，不放在循环内
    def _save_history(self, messages: list[dict]):
        """
           持久化两件事：
           1) history_messages.json  —— 全量会话日志（人类可读，审计用）
           2) memory_state.json      —— 摘要状态快照（程序恢复用，含 _history + _summarized_count + _current_summary）
           两者独立 try/except，互不影响：一个失败不影响另一个落盘。
       """
        file_path = get_abs_path("history_messages/history_messages.json")
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False)
            logger.info("会话记录保存成功")

        except Exception as e:
            logger.error(f"错误：保存历史记录失败，错误信息 {str(e)}")

        # 2. 新增：摘要状态快照（程序恢复用）
        if hasattr(self.base_memory, "save_state"):
            state_path = get_abs_path("history_messages/memory_state.json")
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            self.base_memory.save_state(state_path)

    def _build_retrieval_text(self, msg_list: List[Dict]) -> str:
        """
        将当前上下文消息数组，压缩为检索文本，用于工具向量召回
        只提取role + content；tool_calls结构不放入检索文本
        """
        chunks = []
        for msg in msg_list:
            role = msg.get("role", "")
            content = msg.get("content", "") or ""
            if not content.strip():
                continue
            chunks.append(f"[{role}] {content}")
        full_text = "\n".join(chunks)
        # 限制检索文本最多2000字符，防止向量模型超长
        max_retrieval_char = 2000
        if len(full_text) > max_retrieval_char:
            # 保留尾部最新内容（最近对话、最新tool结果最重要）
            full_text = full_text[-max_retrieval_char:]
        return full_text

    def run(self, query: str):

        # 将本次用户提问加入记忆
        # 写入用户问题，同时写入短期和长期记忆缓冲
        # 用户问题三路写入
        self._record({"role": "user", "content": query})

        # ========== 新增：对话开头召回长期记忆 ==========
        ltm = self.long_term_memory.build_memory_context(query)

        start = time.time()
        # 核心循环，带有参数校验和自纠错
        for i in range(self.max_cycle):
            # 【每轮重新获取】经过token截断+完整性校验的上下文
            current_context = self.base_memory.get_context()

            # ========== 新增：把长期记忆插入上下文 ==========
            # 插在system之后、摘要之前，保证模型优先看到相关历史记忆
            if ltm:
                for idx, msg in enumerate(current_context):
                    if msg["role"] == "system":
                        current_context.insert(idx + 1, ltm)
                        break

            logger.info(f"===== 第 {i+1} 轮思考开始，传给模型窗口上下文共 {len(current_context)} 条消息 =====")

            # ✅修复：每一轮循环内部动态做工具召回，根据当前用户query和上下文重新获取可用工具
            search_context = self._build_retrieval_text(current_context)
            relevant_tool_names = self.tool_retriever.retriever(search_context)
            tool_schema = build_tool_schema(relevant_tool_names)

            response = self.client.chat_completion(
                model=self.model,
                messages=current_context,
                tool_schema=tool_schema
            )

            assistant_msg = response.choices[0].message
            logger.info("模型回复")
            # 模型assistant消息写入滑动窗口全量历史
            # 新增：每轮把模型回复写入长期记忆缓冲
            # assistant 三路写入
            self._record(assistant_msg.model_dump())

            # 若没有调用工具，则直接返回结果，任务结束，执行持久化落盘
            if not assistant_msg.tool_calls:
                logger.info("✅ 模型给出最终答案，结束循环")
                logger.info(f"任务完成，总轮数：{i + 1}，总耗时：{time.time() - start:.2f}s")
                # 修正：对话结束强制flush落盘
                # 不需要再add全量history，每轮已经增量加过了
                if hasattr(self.base_memory, "maybe_summary"):
                    self.base_memory.maybe_summary()
                self.long_term_memory.flush()
                self._save_history(self._session_log)  # ← 写 session_log，不是 base_memory.history
                return assistant_msg.content

            # 并行执行工具，工具结果写入滑动窗口
            tool_results = run_tools_parallel(assistant_msg.tool_calls)
            for tool_call in assistant_msg.tool_calls:
                tool_id = tool_call.id
                tool_result = tool_results[tool_id]
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": tool_result
                }
                # 工具结果写入短期记忆
                # 新增：每轮把工具结果也写入长期记忆缓冲
                self._record(tool_msg)
            # --- 删除原来循环内部频繁的_save_history，不在每轮工具执行后写磁盘 ---
            if hasattr(self.base_memory, "maybe_summary"):
                self.base_memory.maybe_summary()
        # 超出最大轮次，任务终止
        fail_reply = "抱歉，多次尝试后仍未得出结果，请换个方式提问。"
        # 将兜底失败应答作为assistant消息写入记忆，补齐对话回合
        self._record({"role": "assistant", "content": fail_reply})
        logger.info(f"任务失败，总轮数：{self.max_cycle}，总耗时：{time.time() - start:.2f}s，超出最大轮次，结束循环")
        self.long_term_memory.flush()  # 结束前强制落盘
        self._save_history(self._session_log)
        return fail_reply
