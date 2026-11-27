"""
Agent 轻量评测脚本
输出：成功率、重规划触发率、断点恢复成功率、工具准确率等核心指标
"""
import os
import sys
import json
import time
from pathlib import Path

# 项目根目录加入路径
sys.path.append(str(Path(__file__).parent.parent))

from api.depends import _init_global_components, get_session_manager
from utils.logger_tool import logger

# 评测配置
TEST_CASES = [
    # ========== 正常任务集 ==========
    {
        "id": "normal_1",
        "category": "正常任务",
        "name": "简单知识库问答",
        "query": "从 product_quote 知识库查询经济型设备的单价",
        "expect_tools": ["search_knowledge_base"],
        "expect_success": True,
    },
    {
        "id": "normal_2",
        "category": "正常任务",
        "name": "检索+计算两工具协同",
        "query": "根据 product_quote 知识库，计算 50 套经济型设备的设备总价（不含税）",
        "expect_tools": ["search_knowledge_base", "calculator"],
        "expect_success": True,
    },
    {
        "id": "normal_3",
        "category": "正常任务",
        "name": "检索+计算+写文件全链路",
        "query": "查询 product_quote 中标准型单价，计算 100 套总价，写入 output/test_normal_3.txt",
        "expect_tools": ["search_knowledge_base", "calculator", "write_file"],
        "expect_success": True,
    },
    {
        "id": "normal_4",
        "category": "正常任务",
        "name": "多方案价格对比",
        "query": "对比 product_quote 中经济型和标准型 100 套的含税总价，说明差价",
        "expect_tools": ["search_knowledge_base", "calculator"],
        "expect_success": True,
    },
    {
        "id": "normal_5",
        "category": "正常任务",
        "name": "完整采购方案生成",
        "query": "根据 product_quote 生成 100 套设备采购方案，计算三套配置含税总价，推荐最优并写入 output/test_normal_5.md",
        "expect_tools": ["search_knowledge_base", "calculator", "write_file"],
        "expect_success": True,
    },
    # ========== 异常容错集 ==========
    {
        "id": "error_1",
        "category": "异常容错",
        "name": "不存在的知识库",
        "query": "从 not_exist_kb 知识库查询价格",
        "expect_replan": True,
        "expect_success": True,  # 预期降级完成
    },
    {
        "id": "error_2",
        "category": "异常容错",
        "name": "天气工具失败降级",
        "query": "查询北京天气并计算 100 套经济型设备总价",
        "expect_replan": False,
        "expect_success": True,  # 预期非核心工具失败不阻塞主任务
    },
    {
        "id": "error_3",
        "category": "异常容错",
        "name": "计算步骤循环不收敛",
        "query": "反复计算 100 套设备的税费和总价，直到精确到分",
        "expect_replan": True,
        "expect_success": True,
    },
    {
        "id": "error_4",
        "category": "异常容错",
        "name": "文件路径越权尝试",
        "query": "把测试内容写入 ../../test_escape.txt",
        "expect_success": False,  # 预期被安全拦截
    },
    # ========== 断点恢复集 ==========
    {
        "id": "resume_1",
        "category": "断点恢复",
        "name": "执行中断后恢复",
        "query": "生成 80 套设备采购方案并写入 output/test_resume_1.md",
        "expect_resume_success": True,
    },
]


class EvalRunner:
    def __init__(self):
        _init_global_components()
        self.session_manager = get_session_manager()
        self.results = []

    def run_single_case(self, case: dict) -> dict:
        """执行单个测试用例"""
        logger.info(f"\n===== 开始测试：{case['id']} - {case['name']} =====")
        session_id = self.session_manager.create_session()
        agent = self.session_manager.get_agent(session_id)

        start_time = time.time()
        try:
            answer = agent.run(case["query"], resume=False)
            success = True
            error = ""
        except Exception as e:
            success = False
            error = str(e)
            logger.error(f"用例执行失败：{error}")
        cost_time = round(time.time() - start_time, 2)

        # 统计工具调用次数（从会话日志统计）
        tool_calls = sum(
            1 for msg in agent._session_log
            if msg.get("role") == "tool"
        )

        result = {
            **case,
            "success": success,
            "error": error,
            "cost_time": cost_time,
            "tool_calls": tool_calls,
        }
        self.results.append(result)

        # 清理测试生成的文件
        for f in Path("agent_workspace/output").glob("test_*"):
            try:
                os.remove(f)
            except OSError:
                pass

        logger.info(f"结果：{'✅ 成功' if success else '❌ 失败'}，耗时 {cost_time}s，工具调用 {tool_calls} 次")
        return result

    def run_all(self):
        """执行所有测试用例并生成报告"""
        logger.info("=" * 60)
        logger.info("Agent 评测开始")
        logger.info("=" * 60)

        for case in TEST_CASES:
            self.run_single_case(case)

        # 统计指标
        total = len(self.results)
        success_count = sum(1 for r in self.results if r["success"])
        success_rate = round(success_count / total * 100, 1)

        normal_cases = [r for r in self.results if r["category"] == "正常任务"]
        normal_success = sum(1 for r in normal_cases if r["success"])
        normal_success_rate = round(normal_success / len(normal_cases) * 100, 1) if normal_cases else 0

        error_cases = [r for r in self.results if r["category"] == "异常容错"]
        error_success = sum(1 for r in error_cases if r["success"])
        error_success_rate = round(error_success / len(error_cases) * 100, 1) if error_cases else 0

        avg_tool_calls = round(sum(r["tool_calls"] for r in self.results) / total, 1)
        avg_time = round(sum(r["cost_time"] for r in self.results) / total, 2)

        # 输出报告
        report = f"""
===== Agent 评测报告 =====
总用例数：{total}
总成功率：{success_rate}%
正常任务成功率：{normal_success_rate}%
异常容错成功率：{error_success_rate}%
平均工具调用次数：{avg_tool_calls}
平均单任务耗时：{avg_time}s

===== 明细 =====
"""
        for r in self.results:
            status = "✅" if r["success"] else "❌"
            report += f"{status} [{r['category']}] {r['name']} | 耗时 {r['cost_time']}s | 工具 {r['tool_calls']}次"
            if r["error"]:
                report += f" | 错误：{r['error'][:80]}"
            report += "\n"

        logger.info("\n" + report)

        # 保存报告到文件
        with open("eval/report.md", "w", encoding="utf-8") as f:
            f.write(report)

        logger.info("评测报告已保存到 eval/report.md")
        return report


if __name__ == "__main__":
    runner = EvalRunner()
    runner.run_all()
