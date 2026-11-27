from client.llm_client import LLMClient
from tools.tool_retriever import ToolRetriever
from agent_code._008agent import Agent
from tools.tool_registry import tool_registry
from utils.config_tool import agent_conf
from utils.logger_tool import logger
from memory.summary_memory import SummaryMemory
from pydantic_schemas.memory_schemas import MemoryConfig
from memory.long_term_memory import LongTermMemory
from memory.memory_policy import MemoryPolicyConfig


def main(query: str):
    llm_client = LLMClient()
    logger.info("LLMClient 初始化完成")
    tool_retriever = ToolRetriever(tool_registry)
    logger.info("ToolRetriever 工具索引构建完成")
    memory_config = MemoryConfig()
    memory = SummaryMemory(llm_client=llm_client,
                           chat_model=agent_conf["chat_model"],
                           config=memory_config,
                           policy_config=None
    )
    logger.info("SummaryMemory 记忆模块初始化完成")
    memory_policy_config = MemoryPolicyConfig()
    long_term_memory = LongTermMemory(policy_config=memory_policy_config)
    logger.info("LongTermMemory 持久化记忆模块初始化完成")
    agent = Agent(llm_client=llm_client,
                  tool_retriever=tool_retriever,
                  chat_model=agent_conf["chat_model"],
                  base_memory=memory,
                  long_term_memory=long_term_memory,
    )
    logger.info("Agent 初始化完成")
    response = agent.run(query)
    return response
if __name__ == "__main__":
    # res1 = main("我叫糖葫芦，现在我有一个问题，在我们国内，那些地方的冬天不下雪")
    # print(res1)
    # print("-"*30)
    # res2 = main("我喜欢千仞雪，她是斗罗大陆中的角色，千仞雪的结局是什么")
    # print(res2)
    # print("-"*30)
    res3 = main("你发现了吗，我们一直在强调不要以射精为目的，但是我又一直在向你询问如何能够自然射精，这是否有些矛盾，继续下去，我想知道和学习更多知识，那么你还可以继续教我一些知识吗，我希望更加深入的学习和了解更多内容")
    print(res3)
    print("-"*30)
    # res4 = main("我喜欢宁荣荣，她是斗罗大陆中的角色，她的结局是什么")
    # print(res4)
    # print("-"*30)
    # res5 = main("查一下郑州金水区今天的天气")
    # print(res5)
    # print("-" * 30)