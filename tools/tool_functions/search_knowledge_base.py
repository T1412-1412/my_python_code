# 导入知识库管理器（全局单例）
from rag_knowledge.kb_singleton import get_global_kb_manager

# 全局单例，避免重复初始化
_kb_manager = KnowledgeBaseManager()

def search_knowledge_base(kb_name: str, query: str, top_k: int = 3) -> str:
    """
    知识库检索工具函数：从指定知识库中查找相关内容
    返回格式化后的检索结果字符串，直接传给大模型
    """
    results = _kb_manager.search(kb_name=kb_name, query=query, top_k=top_k)
    if not results:
        return f"在知识库【{kb_name}】中未找到相关内容"

    # 格式化成大模型容易阅读的样式
    text = f"从知识库【{kb_name}】中检索到以下相关内容：\n"
    for i, item in enumerate(results, 1):
        text += f"\n[{i}] 来源：{item['source']}（相似度：{item['score']}）\n"
        text += f"内容：{item['content']}\n"
    return text.strip()
