"""
API v1版本路由
"""
from fastapi import FastAPI, APIRouter
from app.api.v1 import filter, pods, bind, network, task_planning, resource_status, scheduler, route_info
from app.core.config import settings

# 所有路由模块列表
api_modules = [filter, pods, bind, network, task_planning, scheduler, route_info]

def register_routers(app: FastAPI, prefix: str = "") -> None:
    """
    自动注册所有API路由
    
    Args:
        app: FastAPI应用实例
        prefix: 路由前缀，默认使用settings.API_V1_STR
    """
    # 使用传入的前缀或默认前缀
    if not prefix:
        prefix = settings.API_V1_STR
    
    # 创建主路由器
    main_router = APIRouter()
    
    # 注册所有模块的路由器
    for module in api_modules:
        if hasattr(module, 'router'):
            main_router.include_router(module.router)
    
    # 注册资源状态服务路由
    main_router.include_router(resource_status.router)
    
    # 将主路由器挂载到应用
    app.include_router(main_router, prefix=prefix) 