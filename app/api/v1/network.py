"""网络信息API路由模块"""
from typing import Dict
from fastapi import APIRouter, HTTPException, Depends
from loguru import logger

from app.schemas.network import NetworkReport
from app.services.network_service import NetworkService

router = APIRouter(tags=["network"], prefix="/network")


def get_network_service():
    """获取网络服务实例
    
    FastAPI依赖注入
    
    Returns:
        NetworkService: 网络服务实例
    """
    # 这个网络服务应该在应用启动时被初始化和注入
    # 这里我们从应用状态中获取，具体实现取决于您的应用状态管理方式
    from app.core.app_state import get_network_service as get_service
    network_service = get_service()
    if network_service is None:
        raise HTTPException(status_code=503, detail="网络信息服务未初始化")
    return network_service


@router.get("/info", summary="获取网络信息")
async def get_network_info(
    source_node: str = None,
    target_node: str = None,
    network_service: NetworkService = Depends(get_network_service)
) -> Dict:
    """获取网络信息
    
    如果指定了源节点和目标节点，返回它们之间的网络信息
    否则返回所有已收集的网络信息
    
    Args:
        source_node: 源节点，可选
        target_node: 目标节点，可选
        network_service: 网络服务实例，自动注入
        
    Returns:
        Dict: 网络信息
    """
    try:
        if source_node and target_node:
            # 获取特定节点间的网络信息
            network_info = network_service.get_network_info(source_node, target_node)
            if network_info:
                return {
                    "source": source_node,
                    "target": target_node,
                    "info": {
                        "latency": network_info.latency,
                        "bandwidth": network_info.bandwidth
                    }
                }
            else:
                return {
                    "source": source_node,
                    "target": target_node,
                    "info": None
                }
        else:
            # 获取所有网络信息
            return {
                "network_info": network_service.get_all_network_info()
            }
    except Exception as e:
        logger.error(f"获取网络信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取网络信息失败: {str(e)}")


@router.post("/test", summary="测试网络信息报告解析")
async def test_network_report(
    report: NetworkReport,
    network_service: NetworkService = Depends(get_network_service)
) -> Dict:
    """测试网络信息报告解析
    
    提交网络信息报告进行测试，将报告存储并返回结果
    
    Args:
        report: 网络信息报告
        network_service: 网络服务实例，自动注入
        
    Returns:
        Dict: 处理结果
    """
    try:
        # 处理网络信息报告
        network_service.network_info_store.update_network_info(report)
        
        return {
            "status": "success",
            "message": f"成功解析网络信息报告，报告ID: {report.report_id}，目的节点数量: {len(report.destinations)}",
            "data": {
                "report_id": report.report_id,
                "source": report.mirror_repository_host_name,
                "destinations": len(report.destinations)
            }
        }
    except Exception as e:
        logger.error(f"测试网络信息报告失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"测试网络信息报告失败: {str(e)}") 