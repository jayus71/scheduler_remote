"""Pod管理API路由"""
from loguru import logger
from typing import Dict

from fastapi import APIRouter, HTTPException, Path

from app.schemas.pods import Pod
from app.services.pods_service import PodsService
from app.api.v1.utils import handle_k8s_error

# 创建路由器
router = APIRouter(tags=["pods"])
pods_service = PodsService()


@router.post("/pods/{namespace}", response_model=Dict)
async def create_pod(
        pod: Pod,
        namespace: str = Path(..., description="命名空间"),
) -> Dict:
    """创建Pod
    
    Args:
        pod: Pod配置
        namespace: 命名空间
        
    Returns:
        Dict: 创建的Pod信息
    """
    try:
        # 去除可能存在的前后空格
        namespace = namespace.strip()
        if not namespace:
            raise HTTPException(status_code=400, detail="命名空间不能为空")

        return pods_service.create_pod(namespace, pod)
    except Exception as e:
        handle_k8s_error(e)


@router.delete("/pods/{namespace}/{pod_name}", response_model=Dict)
async def delete_pod(
        namespace: str = Path(..., description="命名空间"),
        pod_name: str = Path(..., description="Pod名称"),
) -> Dict:
    """删除Pod
    
    Args:
        namespace: 命名空间
        pod_name: Pod名称
        
    Returns:
        Dict: 操作结果信息
    """
    try:
        # 去除可能存在的前后空格
        namespace = namespace.strip()
        pod_name = pod_name.strip()

        # 验证参数
        if not namespace:
            raise HTTPException(status_code=400, detail="命名空间不能为空")
        if not pod_name:
            raise HTTPException(status_code=400, detail="Pod名称不能为空")

        pods_service.delete_pod(namespace, pod_name)
        return {"message": f"Pod {pod_name} 已成功删除"}
    except Exception as e:
        handle_k8s_error(e)


@router.get("/pods/{namespace}/{pod_name}", response_model=Dict)
async def get_pod_status(
        namespace: str = Path(..., description="命名空间"),
        pod_name: str = Path(..., description="Pod名称"),
) -> Dict:
    """获取Pod状态
    
    Args:
        namespace: 命名空间
        pod_name: Pod名称
        
    Returns:
        Dict: Pod状态信息
    """
    try:
        # 去除可能存在的前后空格
        namespace = namespace.strip()
        pod_name = pod_name.strip()

        # 验证参数
        if not namespace:
            raise HTTPException(status_code=400, detail="命名空间不能为空")
        if not pod_name:
            raise HTTPException(status_code=400, detail="Pod名称不能为空")

        return pods_service.get_pod_status(namespace, pod_name)
    except Exception as e:
        handle_k8s_error(e)
        # 这行代码永远不会执行，因为handle_k8s_error总是会抛出异常
        # 但为了类型检查，我们需要返回一个Dict
        return {"error": "未知错误"}
