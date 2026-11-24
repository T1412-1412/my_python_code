"""
摘要生成器
作用：封装所有和摘要生成相关的逻辑（prompt构建，LLM调用，增量摘要，错误降级）
为什么独立：摘要策略可以单独替换（比如换小模型做摘要、换结构化摘要），不影响记忆存储逻辑
"""

import time
from typing import List, Dict, Optional
from utils.logger_tool import logger
from utils.count_tokens import count_tokens
from pydantic_schemas.memory_schemas import SummaryResult


class Summarizer:
    def __init__(self, llm_client, chat_model: str, max_summary_tokens: int = 500):
        self.chat_model = chat_model
        self.max_summary_tokens = max_summary_tokens
        self.llm_client = llm_client    # 复用现有的LLM客户端，和Agent用同一个

    # 核心摘要Prompt：工业级经过优化，明确要求保留关键信息、丢弃细节
    _SUMMARY_PROMPT = """
    你是一个对话历史摘要专家，需要把给定的对话历史压缩成简洁的摘要，严格遵守以下规则：
1. 必须保留的核心信息：用户的核心需求、所有约束条件、已完成的步骤、关键工具调用结果、未解决的问题
2. 必须丢弃的信息：重复内容、无关闲聊、工具调用的冗余细节、模型的思考过程
3. 摘要要连贯，能让没看过原始对话的人快速理解前因后果
4. 不要输出任何解释，直接输出摘要内容
5. 摘要长度不要超过{max_tokens}字

如果提供了「已有摘要」，请在已有摘要的基础上，结合「新增对话」更新摘要，不要重复已有内容。
已有摘要：
{existing_summary}

新增对话：
{new_messages}

更新后的摘要：
""".strip()

    def _build_messages_for_summary(self, existing_summary: str, new_messages: List[Dict]) -> List[Dict]:
        """构建送给摘要模型的消息列表"""
        # 把新增的对话转换成可读文本，工具调用和结果要成对保留
        messages_text = []
        for msg in new_messages:
            if msg["role"] == "user":
                messages_text.append(f"用户：{msg.get('content', '')}")
            elif msg["role"] == "assistant":
                content = msg.get("content", "") or ""
                if msg.get("tool_calls"):
                    # 工具调用只保留工具名和参数，不保留冗余的id等字段
                    for tc in msg["tool_calls"]:
                        content += f"\n调用工具：{tc['function']['name']}，参数：{tc['function']['arguments']}"
                messages_text.append(f"助手：{content}")
            elif msg["role"] == "tool":
                messages_text.append(f"工具返回结果：{msg.get('content', '')}")
        messages_str = "\n".join(messages_text)

        prompt = self._SUMMARY_PROMPT.format(
            max_tokens=self.max_summary_tokens,
            existing_summary=existing_summary if existing_summary else "无",
            new_messages=messages_str
        )
        return [{"role": "user", "content": prompt}]

    def generate_summary(self, existing_summary: str, new_messages: List[Dict]) -> Optional[SummaryResult]:
        """
        生成增量摘要
        :param existing_summary: 之前已经生成的摘要
        :param new_messages: 新增的需要被压缩进摘要的消息
        :return: 摘要结果，失败返回None（上层会降级，返回滑动窗口）
        """
        if not new_messages:
            return None
        try:
            logger.info(f"开始生成摘要，新增待压缩消息{len(new_messages)}条")
            summary_messages = self._build_messages_for_summary(existing_summary, new_messages)
            # 调用LLM生成摘要，不用工具调用，纯文本输出
            response = self.llm_client.chat_completion(
                model = self.chat_model,
                messages = summary_messages,
                tool_schema = None  # 摘要不需要调用工具
            )
            raw = response.choices[0].message.content
            if not raw or not raw.strip():
                logger.warning("摘要返回为空，降级为滑动窗口")
                return None
            summary_text = raw.strip()
            # 结构化返回结果
            result = SummaryResult(
                summary_text=summary_text,
                covered_message_count=len(new_messages),
                total_tokens=count_tokens([{"role": "assistant", "content": summary_text}]),
                created_at=time.time()
            )
            return result
        except Exception as e:
            # 工业级容错：摘要生成失败不崩溃，打日志返回None，上层降级为滑动窗口
            logger.error(f"摘要生成失败，错误信息：{str(e)}，将降级为滑动窗口模式")
            return None