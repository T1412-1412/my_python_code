"""
为规划层提供统一的数据结构，同时增加状态机能力
- 运行时模型：Step/Plan（代码内部维护状态，LLM不可修改）
- 输入契约：StepContract/PlanContract（LLM输出必须严格符合，校验用）
"""
import re
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from utils.logger_tool import logger


# ========== 状态枚举 ==========
# 单步状态
class StepStatus(str, Enum):
    PENDING = "pending"         # 待执行
    IN_PROGRESS = "in_progress" # 执行中
    DONE = "done"               # 已完成
    FAILED = "failed"           # 已失败


# 计划状态
class PlanStatus(str, Enum):
    INITIALIZED = "initialized" # 已初始化
    EXECUTING = "executing"     # 执行中
    REPLANNING = "replanning"  # ★ 新增 重规划中
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"           # 已失败


# ========== 运行时状态模型（代码内部使用，LLM无权修改，新增状态机能力） ==========
@dataclass
class Step:
    description: str
    result: Optional[str] = None
    status: StepStatus = field(default=StepStatus.PENDING)
    depends_on: list[str] = field(default_factory=list)
    step_id: str = field(default_factory=lambda: f"step_{uuid.uuid4().hex}")
    error: Optional[str] = None
    retry_count: int = 0

    # ========== 新增：Step 状态机核心 ==========
    # 合法迁移规则表：{当前状态: {事件: 目标状态}}
    _TRANSITIONS = {
        StepStatus.PENDING: {"start": StepStatus.IN_PROGRESS},
        StepStatus.IN_PROGRESS: {
            "succeed": StepStatus.DONE,
            "fail": StepStatus.FAILED
        },
        StepStatus.FAILED: {"reset": StepStatus.PENDING}
    }

    def transition(self, event: str, error_msg: str = None) -> None:
        """
        唯一状态变更入口：触发事件，自动校验合法性
        非法跳转直接抛异常，绝不静默修改
        """
        allowed = self._TRANSITIONS.get(self.status, {})
        if event not in allowed:
            raise ValueError(
                f"步骤[{self.description}]非法状态跳转："
                f"当前状态={self.status.value}，触发事件={event}，不允许"
            )

        old_status = self.status
        self.status = allowed[event]

        # 自动记录错误信息
        if event == "fail" and error_msg:
            self.error = error_msg

        logger.debug(f"步骤状态变更：{old_status.value} → {self.status.value}（事件：{event}）")


@dataclass
class Plan:
    goal: str
    steps: list[Step]
    status: PlanStatus = field(default=PlanStatus.INITIALIZED)
    replan_count: int = 0
    final_answer: Optional[str] = None
    is_fallback: bool = False

    # 计划自身的查询方法，外部不用写过滤逻辑
    def get_step(self, step_id: str) -> Optional[Step]:
        return next((s for s in self.steps if s.step_id == step_id), None)

    def get_ready_step(self) -> List[Step]:
        """
        判断哪些步骤现在可以执行，一个步骤“可以执行”需要同时满足两个条件：
        1. 步骤的状态是PENDING 2. 步骤的依赖关系已经完成，依赖为空，all()会返回True
        """
        ready = []
        for s in self.steps:
            if s.status != StepStatus.PENDING:
                continue
            if all(self.get_step(d) and self.get_step(d).status == StepStatus.DONE for d in s.depends_on):
                ready.append(s)
        return ready

    def all_done(self) -> bool:
        return all(s.status == StepStatus.DONE for s in self.steps)

    def get_failed_step(self) -> Optional[Step]:
        return next((s for s in self.steps if s.status == StepStatus.FAILED), None)

    # 重规划次数是否耗尽
    def is_replan_exhausted(self, max_replan: int) -> bool:
        return self.replan_count >= max_replan

    # ========== 新增：Plan 状态机核心 ==========
    _TRANSITIONS = {
        PlanStatus.INITIALIZED: {"start_exec": PlanStatus.EXECUTING},
        PlanStatus.EXECUTING: {
            "complete": PlanStatus.COMPLETED,
            "fail": PlanStatus.FAILED,
            "start_replan": PlanStatus.REPLANNING,  # ★ 新增
        },
        PlanStatus.REPLANNING: {
            "replan_ok": PlanStatus.EXECUTING,  # ★ 新增
            "replan_fail": PlanStatus.FAILED,  # ★ 新增
        },
        PlanStatus.FAILED: set(),
        PlanStatus.COMPLETED: set(),
    }

    def transition(self, event: str) -> None:
        """Plan 唯一状态变更入口"""
        allowed = self._TRANSITIONS.get(self.status, {})
        if event not in allowed:
            raise ValueError(
                f"计划非法状态跳转：当前状态={self.status.value}，触发事件={event}，不允许"
            )

        old_status = self.status
        self.status = allowed[event]
        logger.debug(f"计划状态变更：{old_status.value} → {self.status.value}（事件：{event}）")


# ========== 公共工具函数 ==========
def clean_llm_json(raw_text: str) -> dict:
    """
    从模型输出里提取JSON对象：去markdown代码块、去掉前后废话
    解析失败直接抛异常，由调用方catch降级
    """
    text = re.sub(r"```(json)?", "", raw_text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("未找到合法JSON")
    return json.loads(text[start:end + 1])


# ========== 新增：LLM 输入契约 Schema（确定性校验边界） ==========
class StepContract(BaseModel):
    """
    单步计划输入契约：LLM只能输出这些字段
    规则：类型/长度/规则不匹配直接校验失败，走兜底
    字段名和Prompt完全对齐：depends_on 存步骤序号（0开始）
    """
    description: str = Field(
        min_length=2,
        max_length=200,
        description="步骤的目标描述"
    )
    depends_on: List[int] = Field(
        default_factory=list,
        description="依赖的步骤序号，从0开始，必须是整数数组"
    )

    @field_validator("depends_on")
    @classmethod
    def no_negative_index(cls, v: List[int]) -> List[int]:
        """守卫：依赖序号不能为负数"""
        for idx in v:
            if idx < 0:
                raise ValueError(f"依赖序号不能为负数：{idx}")
        return v


class PlanContract(BaseModel):
    """
    整体计划输入契约：LLM输出必须严格符合此结构
    守卫：步骤数限制、依赖越界、自依赖全部在校验层拦截
    """
    steps: List[StepContract] = Field(
        min_length=1,
        max_length=5,
        description="计划步骤列表，最少1步，最多5步"
    )

    @field_validator("steps")
    @classmethod
    def check_dependency_valid(cls, steps: List[StepContract]) -> List[StepContract]:
        """守卫：统一检查依赖合法性（越界、自依赖）"""
        total = len(steps)
        for i, step in enumerate(steps):
            for dep_idx in step.depends_on:
                # 自依赖检查
                if dep_idx == i:
                    raise ValueError(f"第{i}步不能依赖自己")
                # 越界检查
                if dep_idx >= total:
                    raise ValueError(
                        f"第{i}步依赖了不存在的第{dep_idx}步，总共有{total}步"
                    )
        return steps


# ========== 新增：Reflector 重规划输入契约 Schema ==========
class ReplanningStepContract(BaseModel):
    """
    重规划单步契约：LLM输出的修正步骤必须符合此结构
    """
    description: str = Field(
        min_length=2,
        max_length=200,
        description="修正后的步骤目标描述"
    )
    depends_on_index: List[int] = Field(
        default_factory=list,
        description="剩余步骤内部的依赖序号，从0开始"
    )

    @field_validator("depends_on_index")
    @classmethod
    def no_negative_index(cls, v: List[int]) -> List[int]:
        for idx in v:
            if idx < 0:
                raise ValueError(f"依赖序号不能为负数：{idx}")
        return v


class ReplanningContract(BaseModel):
    """
    重规划整体输出契约：LLM必须返回步骤列表+失败原因
    守卫：步骤数上限、字段必填，缺字段直接校验失败
    """
    new_remaining_steps: List[ReplanningStepContract] = Field(
        description="修正后的剩余步骤列表"
    )
    reason: str = Field(
        default="",
        max_length=300,
        description="本次重规划的原因说明"
    )

    @field_validator("new_remaining_steps")
    @classmethod
    def step_count_limit(cls, steps: List[ReplanningStepContract]) -> List[ReplanningStepContract]:
        """守卫：剩余步骤最多5步，防止计划膨胀"""
        if len(steps) > 5:
            raise ValueError(f"剩余步骤不能超过5步，当前{len(steps)}步")
        return steps


# ========== 追加到 plan_model.py 末尾：序列化/反序列化能力 ==========
def step_to_dict(step: Step) -> dict:
    """Step 转可序列化字典，枚举转字符串"""
    return {
        "description": step.description,
        "step_id": step.step_id,
        "status": step.status.value,
        "result": step.result,
        "error": step.error,
        "retry_count": step.retry_count,
        "depends_on": step.depends_on.copy(),
    }

def step_from_dict(data: dict) -> Step:
    """从字典重建 Step 对象"""
    return Step(
        description=data["description"],
        step_id=data["step_id"],
        status=StepStatus(data["status"]),
        result=data.get("result"),
        error=data.get("error"),
        retry_count=data.get("retry_count", 0),
        depends_on=data.get("depends_on", []),
    )

def plan_to_dict(plan: Plan) -> dict:
    """Plan 转可序列化字典"""
    return {
        "goal": plan.goal,
        "status": plan.status.value,
        "replan_count": plan.replan_count,
        "final_answer": plan.final_answer,
        "is_fallback": plan.is_fallback,
        "steps": [step_to_dict(s) for s in plan.steps],
    }

def plan_from_dict(data: dict) -> Plan:
    """从字典重建 Plan 对象"""
    steps = [step_from_dict(s) for s in data["steps"]]
    return Plan(
        goal=data["goal"],
        steps=steps,
        status=PlanStatus(data["status"]),
        replan_count=data.get("replan_count", 0),
        final_answer=data.get("final_answer"),
        is_fallback=data.get("is_fallback", False),
    )
