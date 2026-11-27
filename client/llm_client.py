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

    # 流式响应
    def chat_completion_stream(
            self,
            model: str,
            messages: list,
            temperature: float = 0
    ):
        """
        流式调用：逐块返回文本增量
        增加空帧防御，兼容不同模型的流式响应格式
        """
        logger.info("开始流式生成最终答案")
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            logger.error(f"流式生成失败：{e}")
            yield f"\n[生成异常：{str(e)}]"
