"""节点绑定API路由模块"""
from typing import Optional
from loguru import logger
from fastapi import APIRouter, HTTPException, Depends

from app.schemas.bind import BindRequest, BindResponse
from app.services.bind_service import BindService
from app.api.v1.utils import handle_k8s_error
from app.core.app_state import get_scheduling_plan_store

router = APIRouter(tags=["bind"])
bind_service = BindService()


@router.post("/bind", response_model=BindResponse, summary="节点绑定请求")
async def bind_pod_to_node(request: BindRequest) -> BindResponse:
    """
    将Pod绑定到指定节点
    
    接收节点绑定请求，将Pod绑定到指定的节点上，并返回绑定结果
    
    接口标识符: Schedule-Bind-request
    
    Args:
        request: 绑定请求，包含Pod和目标节点信息
        
    Returns:
        BindResponse: 绑定响应，如果绑定失败，包含错误信息
        
    Raises:
        HTTPException: 当绑定过程中发生错误时
    """
    try:
        logger.info(f"收到节点绑定请求: Pod={request.pod_name}, 节点={request.node}")
        
        # 存储原始请求中的节点
        original_requested_node = request.node
        
        # 获取Pod的调度方案
        scheduling_plan = get_scheduling_plan_store().get_plan(request.pod_name)
        if scheduling_plan:
            selected_node = scheduling_plan.get('selected_node')
            
            # 如果调度方案中有选定节点，则使用调度方案中的节点
            if selected_node:
                # 检查绑定请求中的节点是否与调度方案匹配
                if selected_node != original_requested_node:
                    logger.warning(
                        f"绑定请求中的节点({original_requested_node})与调度方案中的选定节点({selected_node})不一致"
                    )
                    logger.info(f"将使用调度方案中的选定节点({selected_node})进行绑定，而不是请求中的节点")
                    # 修改请求中的节点为调度方案中的选定节点
                    request.node = selected_node
                else:
                    logger.info(f"绑定请求中的节点与调度方案中的选定节点一致({selected_node})")
            else:
                logger.warning(f"Pod {request.pod_name} 的调度方案中没有选定节点，将使用请求中的节点({original_requested_node})")
        else:
            logger.warning(f"未找到Pod {request.pod_name} 的调度方案，将使用请求中的节点({original_requested_node})")

        # 执行绑定操作
        error = await bind_service.bind_pod_to_node(request)

        # 如果绑定成功，可以清除该Pod的调度方案
        if not error:
            get_scheduling_plan_store().remove_plan(request.pod_name)
            logger.info(f"已移除Pod {request.pod_name}的调度方案")
            # 记录实际使用的节点
            logger.info(f"Pod {request.pod_name} 成功绑定到节点 {request.node}")

        # 返回响应
        return BindResponse(error=error)

    except Exception as e:
        logger.error(f"节点绑定过程中发生错误: {str(e)}", exc_info=True)
        return BindResponse(error=str(e))
