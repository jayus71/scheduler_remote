"""网络信息API路由"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Depends, Path
from datetime import datetime

from app.services.network_info_service import NetworkInfoService
from app.schemas.network_info import NetworkReport
from app.core.app_state import get_network_info_service

router = APIRouter(prefix="/network-info", tags=["network-info"])


async def get_network_info_service_dependency() -> NetworkInfoService:
    """获取网络信息服务实例的依赖"""
    service = get_network_info_service()
    if not service:
        raise HTTPException(status_code=503, detail="网络信息服务未初始化")
    return service


@router.get("/report/{mirror_repo_host}", response_model=Optional[NetworkReport])
async def get_network_report(
    mirror_repo_host: str = Path(..., description="镜像仓库节点ID"),
    service: NetworkInfoService = Depends(get_network_info_service_dependency)
):
    """
    获取指定镜像仓库节点的最新网络信息报告
    
    Args:
        mirror_repo_host: 镜像仓库节点ID
        service: 网络信息服务实例（通过依赖注入）
        
    Returns:
        Optional[NetworkReport]: 网络信息报告
    """
    report = service.get_latest_report(mirror_repo_host)
    if not report:
        raise HTTPException(status_code=404, detail="未找到指定节点的网络信息报告")
    return report


@router.get("/timestamp/{mirror_repo_host}")
async def get_report_timestamp(
    mirror_repo_host: str = Path(..., description="镜像仓库节点ID"),
    service: NetworkInfoService = Depends(get_network_info_service_dependency)
):
    """
    获取指定镜像仓库节点的最新报告时间
    
    Args:
        mirror_repo_host: 镜像仓库节点ID
        service: 网络信息服务实例（通过依赖注入）
        
    Returns:
        dict: 包含时间戳的字典
    """
    timestamp = service.get_report_timestamp(mirror_repo_host)
    if not timestamp:
        raise HTTPException(status_code=404, detail="未找到指定节点的报告时间")
    return {"timestamp": timestamp} 