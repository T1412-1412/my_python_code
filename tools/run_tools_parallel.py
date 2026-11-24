import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from json import JSONDecodeError
from tools.execute_tool import execute_tool
from tools.validate_tool_args import validate_tool_args
from utils.config_tool import agent_conf
from utils.logger_tool import logger


# 并行工具调用
# 核心实现：用线程池并行执行
def run_tools_parallel(tools_call: list) -> dict:
    # 最终返回值：{tool_call_id: 执行结果字符串}
    results = {}

    # ========== 第一步：先把所有参数校验做完（串行） ==========
    # 用来存放校验通过可以执行的任务
    task_map = {}

    for tc in tools_call:
        tool_name = tc.function.name
        tool_id = tc.id
        try:
            tool_args = json.loads(tc.function.arguments)
        except JSONDecodeError:
            logger.error(f"工具{tool_name}JSON解析错误")
            results[tool_id] = f"工具{tool_name}JSON解析错误"
            continue

        is_valid, valid_args = validate_tool_args(tool_name, tool_args)
        if not is_valid:
            # 参数错了，直接存错误信息，不用进入线程池
            results[tool_id] = valid_args
        else:
            task_map[tool_id] = (tool_name, valid_args)

    # ========== 第二步：把所有合法任务提交给线程池并行执行 ==========
    # 创建线程池，最多同时跑5个任务
    with ThreadPoolExecutor(max_workers=agent_conf["max_workers"]) as executor:
        # 提货字典，把tool_id和future对应起来
        future_to_id = {}

        for tool_id, (tool_name, tool_args) in task_map.items():
            # 提交任务：告诉线程池，去执行run_tool函数，参数是tool_name和valid_args
            # submit() 立刻返回一个future对象，不用等函数跑完
            future = executor.submit(execute_tool, tool_name, tool_args)
            # 记录这个future对应的是哪个tool_id
            future_to_id[future] = tool_id

        # ========== 第三步：收集线程池执行结果 ==========
        # as_completed会迭代所有future，哪个任务先完成，就先产出哪个，它是一个迭代器，会盯着所有提交的任务
        # 哪个任务先执行完成，就先把它的 `future` 返回给你，不用按提交顺序等，效率更高
        for future in as_completed(future_to_id):
            tool_id = future_to_id[future]
            tool_name = task_map[tool_id][0]
            try:
                # 获取任务的执行结果
                # 如果任务执行中抛了异常，这里会重新抛出来
                tool_result = future.result()
                logger.info(f"工具{tool_name}执行结果：{tool_result}")
                results[tool_id] = tool_result
            except Exception as e:
                logger.error(f"工具{tool_name}执行出错：{str(e)}")
                results[tool_id] = f"工具{tool_name}执行出错：{str(e)}"
    return results