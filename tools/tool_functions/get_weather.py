import json
import requests

# 天气工具函数
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息
    :param city: 城市名称
    :return: 天气信息
    """
    # 模拟工具持续失败，重试也无法恢复
    raise RuntimeError("天气服务暂时不可用，连接超时")
    # try:
    #     url = f"https://wttr.in/{city}?format=j1"
    #     response = requests.get(url, timeout=30)
    #     weather_data = response.json()
    #     return json.dumps(weather_data, ensure_ascii=False)
    # except Exception as e:
    #     return json.dumps({"error": f"天气查询失败：{str(e)}"}, ensure_ascii=False)