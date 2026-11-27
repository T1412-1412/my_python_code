"""
服务启动入口
运行方式：uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.knowledge_base_routes import router as knowledge_router
from api.session_manager_routes import router as session_router
from contextlib import asynccontextmanager
from api.depends import _init_global_components
from utils.logger_tool import logger


# ============ 生命周期函数：管理启动/关闭逻辑 =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化全局组件，避免第一次请求冷启动
    _init_global_components()

    # ========== 关键：打印可直接点击的访问地址 ==========
    # 用 127.0.0.1 本地回环地址，控制台可以直接点击跳转
    # 0.0.0.0 只是监听地址，浏览器无法直接访问
    print("\n" + "=" * 50)
    print("🚀 Agent 服务启动成功")
    print(f"📄 接口文档地址：http://127.0.0.1:8000/docs")
    print(f"💓 健康检查地址：http://127.0.0.1:8000/health")
    print("=" * 50 + "\n")
    # ======================================================

    yield  # 分界线：上面是启动逻辑，下面是关闭逻辑

    # ===== 关闭阶段（停止服务前执行） =====
    logger.info("服务正在关闭，资源清理中...")
    # 后续可在这里加：长期记忆刷盘、会话持久化等清理逻辑

# ===================== 创建 FastAPI 应用 =====================
app = FastAPI(
    title="智能 Agent 服务 API",
    description="基于 Plan-and-Execute + FSM 的智能 Agent 服务接口文档",
    version="1.0.0",
    lifespan=lifespan,
)

# ===================== 配置 CORS 跨域 =====================
# 允许前端跨域调用，开发阶段全开，生产环境要收紧
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],       # 允许所有HTTP方法
    allow_headers=["*"],       # 允许所有请求头
)

# ===================== 注册路由 =====================
app.include_router(session_router)
app.include_router(knowledge_router)


# ===================== 根路径健康检查 =====================
@app.get("/health", summary="健康检查")
def health_check():
    """服务健康检查接口，用于监控"""
    return {"status": "ok", "service": "agent-service"}


# ============ 生命周期函数：管理启动/关闭逻辑 =================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化全局组件，避免第一次请求冷启动
    _init_global_components()

    # ========== 关键：打印可直接点击的访问地址 ==========
    # 用 127.0.0.1 本地回环地址，控制台可以直接点击跳转
    # 0.0.0.0 只是监听地址，浏览器无法直接访问
    print("\n" + "=" * 50)
    print("🚀 Agent 服务启动成功")
    print(f"📄 接口文档地址：http://127.0.0.1:8000/docs")
    print(f"💓 健康检查地址：http://127.0.0.1:8000/health")
    print("=" * 50 + "\n")
    # ======================================================

    yield  # 分界线：上面是启动逻辑，下面是关闭逻辑

    # ===== 关闭阶段（停止服务前执行） =====
    logger.info("服务正在关闭，资源清理中...")
    # 后续可在这里加：长期记忆刷盘、会话持久化等清理逻辑




if __name__ == "__main__":
    # 也可以直接 python main.py 启动
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发模式自动重启
        log_level="info"
    )
