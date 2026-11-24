"""
调用大模型，使用openai接口
"""

from openai import OpenAI
import os



# 创建大模型实例对象
client = OpenAI(
    api_key=os.getenv("API_KEY_qwen"),
    base_url="https://ws-75ntouay5nt8xgfs.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)

# 调用大模型
response = client.chat.completions.create(
    model="qwen3.7-max",
    messages=[{"role": "user", "content": "你是谁，你能帮我做什么"}],
    stream=True
)

for chunk in response:
    if chunk.choices and chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)

