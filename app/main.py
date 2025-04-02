"""应用入口模块"""
import sys
from pathlib import Path
import asyncio



from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from fastapi_offline import FastAPIOffline
import traceback

# 修改导入方式
from app.api.v1 import register_routers
from app.core.config import settings
from app.core.app_state import (
    init_app_state, 
    shutdown_app_state, 
    manage_services,
    get_network_info_service,
    get_computing_power_service,
    get_scheduling_plan_store
)

# 将项目根目录添加到Python路径
ROOT_DIR = str(Path(__file__).parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    
# 配置日志
logger.info("配置日志系统...")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    使用异步上下文管理器处理应用的启动和关闭事件
    """
    # 启动时执行的代码
    logger.info("星载微服务协同系统启动中...")
    logger.info(f"应用名称: {settings.APP_NAME}")
    logger.info(f"版本: {settings.APP_VERSION}")

    try:
        # 初始化应用状态
        await init_app_state()
        
        # 使用统一的服务管理上下文管理器
        async with manage_services():
            # 等待服务完全初始化
            max_retries = 5
            retry_delay = 1  # 秒
            
            for retry in range(max_retries):
                # 验证服务是否正确初始化
                network_info_service = get_network_info_service()
                computing_power_service = get_computing_power_service()
                scheduling_plan_store = get_scheduling_plan_store()
                
                services_ready = (
                    network_info_service is not None and
                    computing_power_service is not None and
                    scheduling_plan_store is not None
                )
                
                if services_ready:
                    logger.info("所有服务已正确初始化")
                    break
                    
                logger.warning(f"服务初始化不完整，重试中 ({retry+1}/{max_retries})...")
                if network_info_service is None:
                    logger.error("网络信息服务未正确初始化")
                if computing_power_service is None:
                    logger.error("算力服务未正确初始化")
                if scheduling_plan_store is None:
                    logger.error("调度方案存储服务未正确初始化")
                    
                # 等待一段时间后重试
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            
            # 最终检查
            network_info_service = get_network_info_service()
            computing_power_service = get_computing_power_service()
            scheduling_plan_store = get_scheduling_plan_store()
            
            if network_info_service:
                logger.info("网络信息服务已正确初始化")
            else:
                logger.error("网络信息服务未正确初始化")
                
            if computing_power_service:
                logger.info("算力服务已正确初始化")
            else:
                logger.error("算力服务未正确初始化")
                
            if scheduling_plan_store:
                logger.info("调度方案存储服务已正确初始化")
            else:
                logger.error("调度方案存储服务未正确初始化")
            
            logger.info("应用启动完成")
            yield  # 应用运行期间
        
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}")
        sys.exit(1)

    # 关闭时执行的代码
    logger.info("正在关闭应用...")
    try:
        # 关闭应用状态
        await shutdown_app_state()
        logger.info("应用已关闭")
    except Exception as e:
        logger.error(f"应用关闭过程中发生错误: {str(e)}")


# 创建FastAPI应用
app = FastAPIOffline(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url=f"{settings.API_V1_PREFIX}/docs",
    redoc_url=f"{settings.API_V1_PREFIX}/redoc",
    lifespan=lifespan,
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 注册API路由
register_routers(app)

# 请求中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有HTTP请求"""
    logger.info(f"请求: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"响应: {request.method} {request.url.path} - {response.status_code}")
    return response


@app.get("/")
async def root():
    """根路由，返回应用信息"""
    return {
        "app": "星载微服务协同系统",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    全局异常处理
    """
    error_detail = str(exc)
    logger.error(f"全局异常: {error_detail}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": "服务器内部错误",
            "detail": error_detail
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="localhost", port=settings.PORT, reload=settings.DEBUG, 
                ws_ping_interval=20, ws_ping_timeout=20)
