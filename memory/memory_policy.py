"""
记忆读写策略混入类（Mixin）
作用：给所有记忆类统一增加写入过滤/打标/去重、读取排序/配额、淘汰能力
设计模式：Mixin混入，不该变原有记忆类的继承结构，无侵入式增强
工业级对应：横切关注点（AOP），所有记忆模块共用同一套策略
"""
import hashlib
import re
import time
from typing import List, Dict, Optional, Tuple
from utils.logger_tool import logger
from utils.count_tokens import count_tokens
from pydantic_schemas.memory_policy_schema import MemoryLevel, MessageMetadata, MemoryPolicyConfig

class MemoryPolicyMixin:
    """读写策略混入类，继承后自动拥有策略能力，必须和BaseMemory子类一起使用"""

    def _init_policy(self, config: MemoryPolicyConfig):
        """初始化策略，在记忆类的__init__中调用一次即可"""
        self.policy_config = config or MemoryPolicyConfig()
        # 编译敏感正则信息，提升匹配效率
        self._sensitive_regex = [re.compile(p, re.IGNORECASE) for p in self.policy_config.sensitive_patterns]
        # 预编译评级关键词正则（提升匹配速度）
        self._core_regex = re.compile(
            "|".join(re.escape(k) for k in self.policy_config.core_fact_keywords),
            re.IGNORECASE
        )
        self._temp_regex = re.compile(
            "|".join(re.escape(k) for k in self.policy_config.temp_keywords),
            re.IGNORECASE
        )
        # 去重缓存：内存消息的hush，用于快速判断重复消息
        self._dedup_cache = set()

    # 优化 检测 + 脱敏一体
    def _detect_and_mask_sensitive(self, content: str) -> Tuple[bool, str]:
        """检测敏感信息并脱敏，返回(是否命中, 脱敏后的内容)"""
        if not content:
            return False, content
        is_sensitive = False
        for regex in self._sensitive_regex:
            # subn：返回元组(替换后的文本, 替换次数)。只需要扫描文本一次，同时完成匹配判定 + 替换，消除重复扫描。
            new_content, sub_count = regex.subn("***", content)
            if sub_count > 0:   # 判断命中
                is_sensitive = True
                content = new_content # 替换脱敏
        return is_sensitive, content

    # 修改：增加多维度消息等级检测（后续可使用大模型对消息评级）
    def _detect_message_level(self, message: Dict) -> MemoryLevel:
        """
        多维度自动检测消息的信息级别
        规则：角色基准 → 强信号跳级 → 辅助特征修正
        """
        role = message.get("role", "")
        content = message.get("content", "") or ""
        content_lower = content.lower()

        # 1. 角色基准判定
        if role == "system":
            return MemoryLevel.SYSTEM

        # tool 角色特殊处理：出错直接TEMP，正常默认PROCESS
        if role == "tool":
            if "error" in content_lower or "失败" in content or "异常" in content:
                return MemoryLevel.TEMP
            return MemoryLevel.PROCESS

        # user/assistant 初始基准
        base_level = MemoryLevel.PROCESS

        # 2. 强信号：命中核心关键词直接升级
        if self._core_regex.search(content):
            return MemoryLevel.CORE_FACT

        # 3. 强信号：命中临时关键词直接降级
        if self._temp_regex.search(content):
            return MemoryLevel.TEMP

        # 4. 辅助特征修正（边界场景微调）
        # 特征1：包含数字/日期等实体，倾向核心事实
        has_entity = bool(re.search(r"\d{4,}", content))
        # 特征2：疑问句，倾向过程（不是结论）
        is_question = content.strip().endswith("？") or content.strip().endswith("?")

        if has_entity and not is_question:
            # 有实体且不是疑问，升为核心（比基准高一级）
            return MemoryLevel.CORE_FACT

        if is_question:
            # 疑问句保持过程级
            return MemoryLevel.PROCESS

        # 默认返回基准级别
        return base_level

    def _add_metadata(self, message: Dict) -> Dict:
        """给消息打元数据标签，返回带标签的消息副本（不修改原消息）"""
        msg = message.copy()

        # 敏感消息检测，同时打上level标签
        is_sensitive, masked_content = self._detect_and_mask_sensitive(msg.get("content", ""))
        if is_sensitive:
            msg["content"] = masked_content
            level = MemoryLevel.SENSITIVE
        else:
            if msg.get("content"):
                msg["content"] = masked_content
            level = self._detect_message_level(msg)

        # 生成元数据
        metadata = MessageMetadata(
            level=level,
            timestamp=time.time(),
            enable_summarize=level not in [MemoryLevel.SYSTEM, MemoryLevel.TEMP],
            enable_persist=level != MemoryLevel.TEMP or self.policy_config.persist_temp_level
        )
        msg["metadata"] = metadata.model_dump()
        return msg

    def _is_duplicate(self, message: Dict) -> bool:
        """判断消息是否重复，用于写入前去重"""
        if not self.policy_config.enable_dedup:
            # 如果配置关闭去重，直接跳过，不做查重
            return False
        # 用role+content的hash作为去重键
        # 拼接：角色+内容，防止出现相同content但role不同被误判重复
        content = f"{message.get('role', '')}:{message.get('content', '')}"
        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        if content_hash in self._dedup_cache:
            return True
        self._dedup_cache.add(content_hash)
        return False

    def policy_add_message(self, message: Dict) -> Optional[Dict]:
        """
        消息写入前策略处理：敏感信息检测 -> 添加元数据 -> 去重，返回处理后的消息，返回None表示该消息不应存储
        在记忆类的add_message方法最开头调用
        """
        # ★ 关键：先判 SENSITIVE 再判去重，顺序不能反
        # 原因：若先去重，敏感消息内容会被写进 _dedup_cache，
        #       后续相同内容的正常消息会被误判为重复而丢弃
        processed_msg = self._add_metadata(message)
        if processed_msg["metadata"]["level"] == MemoryLevel.SENSITIVE:
            return None
        if self._is_duplicate(processed_msg):
            return None
        return processed_msg

    # 久远历史消息按优先级排序
    def _sort_by_priority(self, context: List[Dict]) -> List[Dict]:
        """
        优化，对上下文窗口进行时序安全排序，
        末尾最近 N 轮对话，切片拿出来，完全不动、不排序，保持原始顺序，防止 Agent 逻辑崩坏。
        久远历史的核心事实前置，更容易被模型读到；方便淘汰策略优先清理低优先级消息
        :param context: 记忆类的上下文窗口
        :return: 排序后的上下文窗口
        """
        keep_rounds = getattr(self.policy_config, "keep_rounds", 6) # 保留最近多少轮原文
        protect_msgs = max(1, keep_rounds*2)    # 保护区消息数量

        # head存放所有的system消息，放在最终上下文的头部
        # dialogue存放其他对话消息
        head, dialogue = [], []
        for msg in context:
            # ✅ 防御：跳过非法元素，避免 None 打穿整个链路
            if not isinstance(msg, dict):
                logger.warning(f"_sort_by_priority 跳过非法消息元素：{type(msg)}")
                continue
            if msg.get("role") == "system":  # 顺便把 ["role"] 改成 .get，更稳
                head.append(msg)
            else:
                dialogue.append(msg)
        # 消息比较少，不排序
        if len(dialogue) <= protect_msgs:
            return head + dialogue
        # 切片，把对话消息切为两大区域（核心！）
        history_zone = dialogue[:-protect_msgs] # 久远历史消息
        protect_zone = dialogue[-protect_msgs:] # 最近N轮对话：保时序
        # 对久远历史消息按优先级排序
        history_zone.sort(key=lambda msg: msg.get("metadata", {}).get("level", MemoryLevel.PROCESS))
        return head + history_zone + protect_zone

    # 分级配额淘汰
    def _find_lowest_priority_index(self, context: List[Dict]) -> Tuple[Optional[int], Optional[int]]:
        """
        遍历消息，找到优先级最低（level 数值最大）的那一条，返回它的下标和 level；
        找不到就返回 (None, None)。它保护 SYSTEM 消息永不淘汰
        """
        worst_idx, worst_level = None, None
        for i, msg in enumerate(context):
            level = msg.get("metadata", {}).get("level", MemoryLevel.PROCESS)
            if level == MemoryLevel.SYSTEM:
                continue
            if worst_idx is None or level > worst_level:
                worst_idx, worst_level = i, level
        return worst_idx, worst_level

    def _strip_metadata(self, context: List[Dict]) -> List[Dict]:
        """剥离元数据，返回纯消息，不要传给大模型（浪费token且可能干扰）"""
        return [{k: v for k, v in msg.items() if k != "metadata"} for msg in context]

    def policy_filter_context(self, context: List[Dict]) -> List[Dict]:
        """
        过滤上下文窗口
        读取时策略处理：1.按优先级排序、2.分级配额淘汰、3.剥离元数据
        在记忆类的get_context方法最后调用，返回最终给大模型的上下文
        """
        # 1. 首先按优先级排序
        sorted_context = self._sort_by_priority(context)

        # 2. 排序后再进行分级配额淘汰
        total_token = count_tokens(sorted_context)
        max_total = sum(self.policy_config.level_token_quota.values())

        while total_token > max_total and len(sorted_context) > 2:
            worst_idx, worst_level = self._find_lowest_priority_index(sorted_context)
            if worst_idx is None:
                break
            removed = sorted_context.pop(worst_idx)
            total_token -= count_tokens([removed])
            logger.debug(
                f"淘汰 level={MemoryLevel(worst_level).name} 消息，剩余token {total_token}"
            )

        # 最后剥离metadata
        return self._strip_metadata(sorted_context)

    def policy_evict(self) -> None:
        """
        淘汰策略：定期调用，清理过期消息
        """
        if not hasattr(self, "_history"):
            return
        original_len = len(self._history)
        self._history = [
            msg for msg in self._history
            if msg.get("metadata", "").get("level", MemoryLevel.PROCESS) != MemoryLevel.TEMP
        ]
        if len(self._history) < original_len:
            logger.debug(f"淘汰策略执行：清理了{original_len - len(self._history)}条临时级消息")
