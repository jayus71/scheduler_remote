"""节点过滤API路由"""
from loguru import logger
from fastapi import APIRouter, HTTPException

from app.schemas.filter import FilterRequest, FilterResponse
from app.services.filter_service import FilterService


# 创建路由器
router = APIRouter(tags=["filter"])
filter_service = FilterService()


@router.post("/filter", response_model=FilterResponse)
async def filter_nodes(request: FilterRequest):
    """
    过滤节点接口
    
    根据Pod规格和节点信息过滤符合条件的节点，并使用优化算法选择最佳节点
    
    Args:
        request: 过滤请求，包含Pod规格和节点列表
        
    Returns:
        过滤响应，包含符合条件的节点列表和不符合条件的节点及原因
    """
    logger.info(f"收到过滤请求: Pod={request.pod.metadata.name}, 节点数量={len(request.nodes)}")

    try:
        # 过滤节点
        filter_result = await filter_service.filter_nodes(request)

        # 构建响应
        response = FilterResponse(
            nodes=filter_result.nodes,
            node_names=filter_result.node_names,
            failed_nodes=filter_result.failed_nodes
        )

        logger.info(
            f"过滤结果: 符合条件的节点数量={len(response.nodes)}, 不符合条件的节点数量={len(response.failed_nodes)}")
        return response

    except Exception as e:
        logger.error(f"过滤节点时发生错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"过滤节点失败: {str(e)}")
