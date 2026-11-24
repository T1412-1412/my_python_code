# 计算器工具函数
import json


def calculator(a: float, b: float, operator: str) -> float|str:
    """
    简易的计算器，计算两个数的和、差、积、商
    :param a: 数字1
    :param b: 数字2
    :param operator: 运算符必须为其中之一：+、-、*、/
    :return: 计算结果，如果运算符无效则返回错误信息
    """
    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        return a * b
    elif operator == "/":
        if b != 0:
            return round(a / b, 2)
        else:
            return json.dumps({"error": "错误：除数不能为零"}, ensure_ascii=False)
    else:
        return json.dumps({"error": "错误：无效的运算符"}, ensure_ascii=False)