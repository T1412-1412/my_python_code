from typing import List, Dict


def validate_context_integrity(context: List[Dict]) -> List[Dict]:
    """
    工业级上下文完整性校验器
    双向校验头部+尾部，修复所有不合法的消息结构
    保证最终输出严格符合模型消息格式协议
    """
    if not context:
        return []

    # ========== 第一步：结构合法性预过滤 ==========
    # 删掉所有结构残缺的消息（缺role、缺content等必填字段）
    valid_context = []
    for msg in context:
        if "role" not in msg:
            continue  # 连角色都没有，直接丢弃
        role = msg["role"]
        # 基础字段校验
        if role in ("user", "system") and "content" not in msg:
            continue
        if role == "tool" and ("tool_call_id" not in msg or "content" not in msg):
            continue
        if role == "assistant":
            # assistant可以没有content（纯工具调用），但不能既没content又没tool_calls
            if "content" not in msg and "tool_calls" not in msg:
                continue
        valid_context.append(msg)

    if not valid_context:
        return []

    # ========== 第二步：头部修复 ==========
    # 规则1：头部不能以tool消息开头（缺少对应的assistant调用）
    while valid_context and valid_context[0]["role"] == "tool":
        valid_context.pop(0)

    # ========== 第三步：尾部修复 ==========
    # 规则2：尾部不能以悬空的tool_calls结尾（缺少对应的tool结果）
    while valid_context and valid_context[-1]["role"] == "assistant":
        last_msg = valid_context[-1]
        # 如果带tool_calls且后面没有tool结果 → 悬空，删除
        if "tool_calls" in last_msg and last_msg["tool_calls"]:
            valid_context.pop()
        else:
            break  # 普通assistant消息，合法，停止

    # ========== 第四步：连续相同角色合并 ==========
    # 规则3：不允许连续两条相同角色的消息（system除外）
    merged_context = []
    for msg in valid_context:
        if not merged_context:
            merged_context.append(msg.copy())
            continue

        last = merged_context[-1]
        # 角色相同且都不是system → 合并
        if msg["role"] == last["role"] and msg["role"] != "system":
            # 合并content
            old_content = last.get("content", "") or ""
            new_content = msg.get("content", "") or ""
            if old_content and new_content:
                last["content"] = old_content + "\n" + new_content
            elif new_content:
                last["content"] = new_content

            # 合并tool_calls（assistant角色）
            if msg["role"] == "assistant" and "tool_calls" in msg:
                if "tool_calls" not in last:
                    last["tool_calls"] = []
                last["tool_calls"].extend(msg["tool_calls"])
        else:
            merged_context.append(msg.copy())

    return merged_context