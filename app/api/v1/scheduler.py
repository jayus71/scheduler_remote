"""调度器API路由模块"""
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Path
from loguru import logger
import asyncio

from app.schemas.priority import PriorityRequest, PriorityResponse, HostPriority
from app.schemas.common import NodeScore
from app.core.app_state import get_network_info_service, get_computing_power_service, get_scheduling_plan_store
from app.services.optimizer import OptimizerService
from app.core.config import settings
from app.services.scheduler import SchedulerService
from app.utils.resource_parser import parse_resource_value

router = APIRouter(tags=["priority"])

# 获取服务实例
network_info_service = get_network_info_service()
computing_power_service = get_computing_power_service()
scheduling_plan_store = get_scheduling_plan_store()


@router.post("/priority", response_model=PriorityResponse)
async def calculate_priority(request: PriorityRequest) -> PriorityResponse:
    """计算节点优先级
    
    接口标识符: Schedule-priority-request
    接收星载微服务协同发送的节点打分请求，等待算力和网络信息上报，然后计算节点优先级
    
    Args:
        request: 优先级计算请求，包含Pod信息和节点列表
        
    Returns:
        PriorityResponse: 节点优先级响应，包含节点优先级列表
    """
    try:
        # 获取最新的服务实例
        current_scheduling_plan_store = get_scheduling_plan_store()
        current_network_info_service = get_network_info_service()
        current_computing_power_service = get_computing_power_service()
        
        # 验证关键服务是否已初始化
        services_status = []
        if not current_scheduling_plan_store:
            services_status.append("调度方案存储服务未初始化")
        if not current_network_info_service:
            services_status.append("网络信息服务未初始化")
        if not current_computing_power_service:
            services_status.append("算力服务未初始化")
            
        if services_status:
            error_msg = "、".join(services_status)
            logger.error(f"计算节点优先级失败: {error_msg}")
            return PriorityResponse(
                hostPriorityList=[],
                error_msg=f"计算节点优先级失败: {error_msg}"
            )
        
        # 在函数内部初始化 optimizer_service
        optimizer_service = OptimizerService(current_scheduling_plan_store)
        
        pod_name = request.pod.metadata.name
        # 等待算力信息和网络信息上报
        logger.info(f"收到节点优先级计算请求，Pod: {pod_name}，等待算力和网络信息上报...")
        
        # 检查服务是否初始化
        computing_power_received = False
        network_info_received = False
        
        # 并行等待算力信息和网络信息，仅当服务实例存在时
        wait_tasks = []
        
        if current_computing_power_service:
            wait_tasks.append(current_computing_power_service.wait_for_report(timeout=settings.UDP_TIMEOUT))
            logger.info("等待算力信息上报...")
        else:
            logger.warning("算力服务未初始化，跳过等待算力信息上报")
            
        if current_network_info_service:
            wait_tasks.append(current_network_info_service.wait_for_report(timeout=settings.UDP_TIMEOUT))
            logger.info("等待网络信息上报...")
        else:
            logger.warning("网络信息服务未初始化，跳过等待网络信息上报")
            
        # 如果有等待任务，则执行
        if wait_tasks:
            try:
                # 使用gather并处理异常
                results = await asyncio.gather(*wait_tasks, return_exceptions=True)
                
                # 处理结果
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.warning(f"等待服务信息时出错: {result}")
                    elif i == 0 and current_computing_power_service:  # 第一个任务是算力服务
                        computing_power_received = result
                        logger.info(f"算力信息接收状态: {computing_power_received}")
                    elif current_network_info_service:  # 第二个任务是网络信息服务
                        network_info_received = result
                        logger.info(f"网络信息接收状态: {network_info_received}")
            except Exception as e:
                # 捕获异步等待过程中的任何异常
                logger.error(f"等待服务信息报告时发生错误: {str(e)}")
        
        # 获取算力信息
        computing_power_info = None
        if request.computing_power_info:
            # 优先使用请求中提供的算力信息
            computing_power_info = request.computing_power_info
            logger.info("使用请求中提供的算力信息")
        elif current_computing_power_service and computing_power_received:
            # 其次使用通过UDP接收的算力信息
            computing_power_info = {}
            for node in request.nodes:
                node_info = current_computing_power_service.get_latest_report(node.name)
                if node_info:
                    # 转换为优化器服务所需的格式
                    computing_power_info[node.name] = {
                        'cpu_load': node_info.cpu_load_percentage / 100 if hasattr(node_info, 'cpu_load_percentage') else 0.5,
                        'gpu_load': node_info.gpu_load_percentage / 100 if hasattr(node_info, 'gpu_load_percentage') else 0.5,
                        'fpga_load': node_info.fpga_load_percentage / 100 if hasattr(node_info, 'fpga_load_percentage') else 0.5
                    }
            logger.info(f"从算力服务获取了 {len(computing_power_info)} 个节点的算力信息")
        else:
            # 如果没有算力信息，创建一个带有默认值的信息
            logger.warning("未收到算力信息，使用默认值")
            computing_power_info = {}
        
        # 获取网络信息
        network_info = None
        if request.network_info:
            # 优先使用请求中的网络信息
            network_info = request.network_info
            logger.info("使用请求中提供的网络信息")
        elif current_network_info_service and network_info_received:
            # 其次使用通过UDP接收的网络信息
            network_info = {}
            # 获取所有镜像仓库的网络信息
            for mirror_repo_host in current_network_info_service.latest_reports.keys():
                report = current_network_info_service.get_latest_report(mirror_repo_host)
                if report:
                    for dest in report.destinations:
                        # 只添加请求中包含的节点的网络信息
                        if any(node.name == dest.dest_host_name for node in request.nodes):
                            network_info[dest.dest_host_name] = {
                                'latency': dest.network_info.latency,
                                'bandwidth': dest.network_info.bandwidth
                            }
            logger.info(f"从网络信息服务获取了 {len(network_info) if network_info else 0} 个节点的网络信息")
        else:
            # 如果没有网络信息，使用默认值
            logger.warning("未收到网络信息，使用默认值")
            network_info = {}
        
        # 准备基础资源信息
        resource_info = {}
        for node in request.nodes:
            try:
                # 计算资源使用率
                alloc_cpu = parse_resource_value(str(node.allocated.cpu)) if node.allocated else 0
                alloc_mem = parse_resource_value(str(node.allocated.memory)) if node.allocated else 0
                allocatable_cpu = parse_resource_value(str(node.allocatable.cpu))
                allocatable_mem = parse_resource_value(str(node.allocatable.memory))
                
                cpu_usage = alloc_cpu / allocatable_cpu if allocatable_cpu > 0 else 0
                mem_usage = alloc_mem / allocatable_mem if allocatable_mem > 0 else 0
                
                resource_info[node.name] = {
                    'cpu': cpu_usage,
                    'memory': mem_usage
                }
                
                # 检查节点是否有GPU资源
                has_gpu = False
                gpu_count = 0
                if hasattr(node.capacity, 'nvidia.com/gpu'):
                    try:
                        gpu_count = int(node.capacity.gpu)
                        has_gpu = gpu_count > 0
                    except (ValueError, TypeError):
                        logger.warning(f"无法解析节点 {node.name} 的GPU容量")
                
                # 如果有算力信息，添加GPU信息
                if computing_power_info and node.name in computing_power_info:
                    computing_power_info[node.name]['has_gpu'] = has_gpu
                    computing_power_info[node.name]['gpu_count'] = gpu_count
                elif node.name not in computing_power_info:
                    # 如果节点没有算力信息，添加默认值
                    if computing_power_info is None:
                        computing_power_info = {}
                        
                    computing_power_info[node.name] = {
                        'cpu_load': 0.5,  # 默认负载50%
                        'gpu_load': 0.5,
                        'fpga_load': 0.5,
                        'has_gpu': has_gpu,
                        'gpu_count': gpu_count
                    }
                
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"处理节点 {node.name} 资源信息时出错: {str(e)}")
                # 使用默认值
                resource_info[node.name] = {
                    'cpu': 0.5,  # 默认使用率50%
                    'memory': 0.5
                }
                
                # 确保节点有算力信息
                if computing_power_info is not None and node.name not in computing_power_info:
                    computing_power_info[node.name] = {
                        'cpu_load': 0.5,
                        'gpu_load': 0.5,
                        'fpga_load': 0.5,
                        'has_gpu': False,
                        'gpu_count': 0
                    }
        
        # 确保至少有一个节点有资源信息
        if not resource_info:
            logger.warning("所有节点的资源信息处理失败，使用默认值")
            for node in request.nodes:
                resource_info[node.name] = {
                    'cpu': 0.5,
                    'memory': 0.5
                }
                
                if computing_power_info is not None and node.name not in computing_power_info:
                    computing_power_info[node.name] = {
                        'cpu_load': 0.5,
                        'gpu_load': 0.5,
                        'fpga_load': 0.5,
                        'has_gpu': False,
                        'gpu_count': 0
                    }
        
        # 调用优化器服务计算节点得分
        logger.info("开始计算节点优先级...")
        scores_result = optimizer_service.calculate_node_scores(
            pod_name=pod_name,
            nodes=[node.name for node in request.nodes],
            resource_info=resource_info,
            computing_power_info=computing_power_info,
            network_info=network_info
        )
        
        logger.info(f"节点优先级计算完成，共有 {len(scores_result.hostPriorityList)} 个节点")
        return scores_result
    except Exception as e:
        logger.error(f"计算节点优先级时出错: {str(e)}", exc_info=True)
        # 返回带有错误信息的响应
        return PriorityResponse(
            hostPriorityList=[],
            error_msg=f"计算节点优先级失败: {str(e)}"
        )
        
@router.get("/plan/{pod_name}", response_model=Dict[str, Any])
async def get_scheduling_plan(
    pod_name: str = Path(..., description="Pod名称")
) -> Dict[str, Any]:
    """获取Pod的调度方案
    
    接口标识符: Schedule-plan-query
    
    Args:
        pod_name: Pod名称
        
    Returns:
        Dict[str, Any]: 调度方案
    """
    # 获取最新的服务实例
    current_scheduling_plan_store = get_scheduling_plan_store()
    if not current_scheduling_plan_store:
        raise HTTPException(
            status_code=500,
            detail="调度方案存储服务未初始化"
        )
    
    plan = current_scheduling_plan_store.get_plan(pod_name)
    if not plan:
        raise HTTPException(
            status_code=404,
            detail=f"未找到Pod {pod_name} 的调度方案"
        )
    return plan 

@router.get("/select/{pod_name}", response_model=Dict[str, Any])
async def select_node_for_pod(
    pod_name: str = Path(..., description="Pod名称")
) -> Dict[str, Any]:
    """为Pod选择最佳节点
    
    接口标识符: Schedule-node-select
    
    根据之前计算的优先级方案为Pod选择最佳节点
    
    Args:
        pod_name: Pod名称
        
    Returns:
        Dict[str, Any]: 包含选定节点或错误信息的响应
    """
    # 获取最新的服务实例
    current_scheduling_plan_store = get_scheduling_plan_store()
    if not current_scheduling_plan_store:
        raise HTTPException(
            status_code=500,
            detail="调度方案存储服务未初始化"
        )
    
    # 在函数内部初始化 scheduler_service
    scheduler_service = SchedulerService(current_scheduling_plan_store)
    
    selected_node, error = await scheduler_service.select_node_for_pod(pod_name)
    
    if error:
        raise HTTPException(
            status_code=404 if "未找到" in error else 500,
            detail=error
        )
        
    return {
        "pod_name": pod_name,
        "selected_node": selected_node
    }

@router.get("/scores/{pod_name}", response_model=List[NodeScore])
async def get_node_scores(
    pod_name: str = Path(..., description="Pod名称")
) -> List[NodeScore]:
    """获取Pod的节点得分列表
    
    接口标识符: Schedule-node-scores
    
    返回之前为Pod计算的节点得分列表
    
    Args:
        pod_name: Pod名称
        
    Returns:
        List[NodeScore]: 节点得分列表
    """
    # 获取最新的服务实例
    current_scheduling_plan_store = get_scheduling_plan_store()
    if not current_scheduling_plan_store:
        raise HTTPException(
            status_code=500,
            detail="调度方案存储服务未初始化"
        )
    
    # 在函数内部初始化 scheduler_service
    scheduler_service = SchedulerService(current_scheduling_plan_store)
    
    scores = await scheduler_service.get_node_scores(pod_name)
    
    if not scores:
        raise HTTPException(
            status_code=404,
            detail=f"未找到Pod {pod_name} 的节点得分信息"
        )
        
    return scores 