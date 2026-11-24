from client.base_client import get_base_client
import time
from utils.logger_tool import logger

class LLMClient:
    def __init__(self):
        self.client = get_base_client()

    # agent调用client.chat.completion.creat()
    def chat_completion(self,
                        model,
                        messages: list,
                        tool_schema: list,
                        tool_choice="auto",
                        temperature=0,
                        stream=False,
                        response_format=None):
        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_schema,
                tool_choice=tool_choice,
                temperature=temperature,
                response_format=response_format,
                stream=stream
            )
            cost = time.time() - start
            logger.info(f"模型调用完成，耗时 {cost:.2f}s")
            return response
        except Exception as e:
            logger.error(f"模型调用异常：{e}", exc_info=True)
            raise
