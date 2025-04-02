"""API工具函数"""
from fastapi import HTTPException
from kubernetes.client import ApiException
from loguru import logger


def handle_k8s_error(e: Exception) -> None:
    """处理Kubernetes API错误
    
    Args:
        e: 异常对象
    
    Raises:
        HTTPException: HTTP错误响应
    """
    if isinstance(e, ApiException):
        if e.status == 404:
            logger.warning(f"资源未找到: {e.reason}")
            raise HTTPException(
                status_code=404,
                detail=str(e.reason)
            )
        elif e.status == 409:
            logger.warning(f"资源冲突: {e.reason}")
            raise HTTPException(
                status_code=409,
                detail="资源冲突"
            )
    logger.error(f"Kubernetes API错误: {str(e)}", exc_info=True)
    raise HTTPException(status_code=500, detail=str(e))
