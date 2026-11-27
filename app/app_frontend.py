"""
Agent 演示前端 - 全程打字机流式版
所有文本（计划、步骤结果、最终答案）均模拟逐字输出，观感统一
"""
import streamlit as st
import requests
import json
import time

API_BASE = "http://127.0.0.1:8000/api/v1"

st.set_page_config(
    page_title="智能任务 Agent 演示",
    page_icon="🤖",
    layout="wide"
)

# ===================== 全局状态初始化 =====================
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_list" not in st.session_state:
    st.session_state.session_list = []

# ===================== 工具函数 =====================
def type_text(placeholder, text: str, speed: float = 0.003):
    """
    模拟打字机逐字输出效果
    placeholder: streamlit 占位组件
    text: 要显示的完整文本
    speed: 每个字的间隔秒数，越小越快
    """
    current = ""
    for char in text:
        current += char
        placeholder.markdown(current + "▌")
        time.sleep(speed)
    placeholder.markdown(current)
    return current

def refresh_session_list():
    try:
        resp = requests.get(f"{API_BASE}/sessions", timeout=5)
        if resp.status_code == 200:
            st.session_state.session_list = resp.json()["data"]["items"]
    except:
        pass

def load_session_history(sid):
    try:
        resp = requests.get(f"{API_BASE}/sessions/{sid}/history", timeout=5)
        if resp.status_code == 200:
            msgs = resp.json()["data"]["messages"]
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]}
                for m in msgs if m["role"] in ("user", "assistant")
            ]
        else:
            st.session_state.messages = []
    except:
        st.session_state.messages = []

def create_new_session():
    try:
        resp = requests.post(f"{API_BASE}/sessions", timeout=5)
        if resp.status_code == 200:
            new_id = resp.json()["data"]["session_id"]
            st.session_state.session_id = new_id
            st.session_state.messages = []
            refresh_session_list()
            st.rerun()
        else:
            st.toast(f"创建失败：{resp.json().get('detail', '未知错误')}", icon="⚠️")
    except Exception as e:
        st.toast(f"创建失败：{str(e)}", icon="⚠️")

# ===================== 侧边栏 =====================
with st.sidebar:
    if st.button("➕ 新建对话", use_container_width=True, type="primary"):
        create_new_session()

    st.divider()
    st.subheader("历史对话")
    if st.button("🔄 刷新列表", use_container_width=True):
        refresh_session_list()
        st.rerun()

    for item in st.session_state.session_list:
        sid = item["session_id"]
        title = f"对话 {sid[:8]}..."
        is_active = (sid == st.session_state.session_id)
        if st.button(
            title,
            key=sid,
            use_container_width=True,
            type="primary" if is_active else "secondary"
        ):
            st.session_state.session_id = sid
            load_session_history(sid)
            st.rerun()

    st.divider()
    with st.expander("📚 知识库管理", expanded=False):
        try:
            kb_resp = requests.get(f"{API_BASE}/knowledge-bases", timeout=5)
            kb_list = [item["name"] for item in kb_resp.json()["data"]["items"]] if kb_resp.status_code == 200 else []
        except:
            kb_list = []

        selected_kb = st.selectbox("选择知识库", kb_list)
        uploaded_file = st.file_uploader("上传文档", type=["txt", "md", "pdf"])

        if st.button("上传", use_container_width=True) and uploaded_file and selected_kb:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                requests.post(
                    f"{API_BASE}/knowledge-bases/{selected_kb}/upload",
                    files=files,
                    timeout=30
                )
                st.success("上传成功")
            except Exception as e:
                st.error(f"上传失败：{str(e)}")

# ===================== 主对话区域 =====================
st.title("🤖 智能任务执行 Agent")
st.caption("Plan-and-Execute 架构 · FSM 状态机 · 断点恢复 · 全流程流式输出")

if not st.session_state.session_id:
    st.info("👈 请先在左侧新建对话，开始执行任务")
    st.stop()

# 历史对话渲染
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 底部输入框
if prompt := st.chat_input("输入任务指令，例如：根据 product_quote 生成 120 套设备采购方案"):
    # 记录用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    # 本轮助手回复
    with chat_container:
        with st.chat_message("assistant"):
            output_placeholder = st.empty()
            full_text = ""
            current_placeholder = output_placeholder

            try:
                with requests.post(
                    f"{API_BASE}/sessions/{st.session_state.session_id}/chat-stream",
                    json={"query": prompt, "resume": True},
                    stream=True,
                    timeout=300
                ) as resp:
                    if resp.status_code != 200:
                        full_text = f"❌ 请求失败：{resp.text}"
                        output_placeholder.markdown(full_text)
                    else:
                        for line in resp.iter_lines(decode_unicode=True):
                            if not line or not line.startswith("data: "):
                                continue

                            try:
                                event = json.loads(line[len("data: "):])
                                event_type = event["type"]
                                event_data = event["data"]
                            except:
                                continue

                            # ========== 1. 状态提示：临时显示 ==========
                            if event_type == "status":
                                status_text = f"\n⏳ {event_data}\n"
                                current_placeholder.markdown(full_text + status_text)

                            # ========== 2. 规划完成：逐字打出计划 ==========
                            elif event_type == "plan":
                                # 拼接计划文本
                                plan_text = "\n**📋 执行计划**\n\n"
                                for s in event_data["steps"]:
                                    plan_text += f"{s['index']}. {s['desc']}\n"
                                plan_text += "\n---\n"

                                # 逐字打出计划
                                current_placeholder.empty()
                                full_text += plan_text
                                current_placeholder = st.empty()
                                type_text(current_placeholder, full_text, speed=0.002)

                            # ========== 3. 单步完成：逐字打出步骤结果 ==========
                            elif event_type == "step":
                                step_text = f"\n✅ 步骤{event_data['index']}完成：{event_data['summary']}\n"
                                full_text += step_text
                                current_placeholder.empty()
                                current_placeholder = st.empty()
                                type_text(current_placeholder, full_text, speed=0.002)

                            # ========== 4. 最终答案：原生逐字追加 ==========
                            elif event_type == "content":
                                full_text += event_data
                                current_placeholder.markdown(full_text + "▌")

                            # ========== 5. 结束 ==========
                            elif event_type == "done":
                                current_placeholder.markdown(full_text)

            except Exception as e:
                full_text += f"\n\n❌ 执行异常：{str(e)}"
                current_placeholder.markdown(full_text)

            # 保存到历史消息
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_text.strip()
            })
