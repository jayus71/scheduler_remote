"""调度器服务模块

提供节点选择和调度功能
"""
import os
from datetime import timedelta
from typing import Dict, List, Tuple, Optional
import random
import numpy as np
from loguru import logger

from app.schemas.filter import FilterRequest, FilterResponse, FilterResult
from app.schemas.common import Pod, Node, NodeScore
from app.services.k8s_service import KubernetesService
from app.services.optimizer import OptimizerService
from app.core.config import settings
from app.utils.resource_parser import parse_resource_value
from app.schemas.priority import HostPriority
from app.services.scheduling_plan_store import SchedulingPlanStore


async def check_node_resources(
        node: Dict,
        cpu_request: int,
        memory_request: int
) -> Tuple[bool, str]:
    """
    检查节点资源是否满足请求
    
    Args:
        node: 节点信息
        cpu_request: CPU请求量（毫核）
        memory_request: 内存请求量（MB）
        
    Returns:
        Tuple[bool, str]: (是否满足, 原因)
    """
    try:
        # 获取节点可分配资源
        allocatable = node.get("status", {}).get("allocatable", {})
        
        # 解析CPU和内存资源
        cpu_capacity = parse_resource_value(allocatable.get("cpu", "0"))
        memory_capacity = parse_resource_value(allocatable.get("memory", "0"))
        
        # 检查CPU资源
        if cpu_capacity < cpu_request:
            return False, f"CPU资源不足: 需要{cpu_request}m, 可用{cpu_capacity}m"
            
        # 检查内存资源
        if memory_capacity < memory_request:
            return False, f"内存资源不足: 需要{memory_request}MB, 可用{memory_capacity}MB"
            
        return True, "资源充足"
    except Exception as e:
        logger.error(f"检查节点资源时出错: {str(e)}")
        return False, f"检查资源时出错: {str(e)}"


class SchedulerService:
    """调度器服务类"""
    
    def __init__(self, plan_store: SchedulingPlanStore):
        """初始化调度器服务
        
        Args:
            plan_store: 调度方案存储服务实例
        """
        self._plan_store = plan_store
        self.k8s_service = KubernetesService()
        # 初始化优化器服务实例，传递调度方案存储服务
        self.optimizer_service = OptimizerService(plan_store)
        # 从settings获取资源权重配置
        self.resource_weights = {
            "cpu": settings.WEIGHT_CPU,
            "memory": settings.WEIGHT_MEMORY
        }
        # 是否启用优化器
        self.enable_optimizer = settings.ENABLE_OPTIMIZER

    async def filter_nodes(self, request: FilterRequest) -> FilterResponse:
        """节点预过滤逻辑
        
        Args:
            request: 过滤请求
            
        Returns:
            FilterResponse: 过滤响应
        """
        response = FilterResponse()
        failed_nodes: Dict[str, str] = {}

        # 如果没有提供节点列表，从Kubernetes获取
        if not request.nodes:
            request.nodes = self.k8s_service.list_nodes()
            logger.info(f"从Kubernetes获取到 {len(request.nodes)} 个节点")

        # 获取容器的资源请求
        container = request.pod.spec.containers[0]  # 当前仅处理第一个容器

        try:
            # 尝试获取资源请求，如果不存在则使用默认值
            cpu_request = parse_resource_value(getattr(container.resources.requests, "cpu", "100m"))
            memory_request = parse_resource_value(getattr(container.resources.requests, "memory", "128Mi"))
        except AttributeError:
            # 如果无法访问resources.requests，使用默认值
            logger.warning("无法获取容器资源请求，使用默认值")
            cpu_request = parse_resource_value("100m")
            memory_request = parse_resource_value("128Mi")

        logger.info(f"Pod资源需求: CPU={cpu_request}m, 内存={memory_request}MB")

        # 如果指定了具体节点
        if request.pod.spec.nodeName:
            target_node = next(
                (node for node in request.nodes if node.name == request.pod.spec.nodeName),
                None
            )
            if not target_node:
                response.error = f"指定的节点 {request.pod.spec.nodeName} 不存在"
                return response

            is_fit, reason = await check_node_resources(
                target_node,
                cpu_request,
                memory_request
            )
            if is_fit:
                node_score = NodeScore(name=target_node.name, score=100)  # 指定节点默认得分为100
                response.nodes = [node_score]
                response.node_names = [target_node.name]
            else:
                failed_nodes[target_node.name] = reason
            response.failed_nodes = failed_nodes
            return response

        # 如果指定了节点选择器
        if request.pod.spec.node_selector:
            logger.info(f"使用节点选择器: {request.pod.spec.node_selector}")
            filtered_nodes = []
            for node in request.nodes:
                # 检查节点标签是否匹配
                if not all(
                        node.labels.get(key) == value
                        for key, value in request.pod.spec.node_selector.items()
                ):
                    failed_nodes[node.name] = "节点标签不匹配"
                    continue

                # 检查资源是否满足
                is_fit, reason = await check_node_resources(
                    node,
                    cpu_request,
                    memory_request
                )
                if is_fit:
                    filtered_nodes.append(node)
                else:
                    failed_nodes[node.name] = reason

            # 使用优化器对节点进行排序
            if self.enable_optimizer and filtered_nodes:
                # 准备节点名称和资源信息
                node_names = [node.name for node in filtered_nodes]
                resource_info = {}
                
                # 构建资源信息字典
                for node in filtered_nodes:
                    try:
                        allocatable = node.status.allocatable
                        cpu_capacity = parse_resource_value(allocatable.cpu)
                        memory_capacity = parse_resource_value(allocatable.memory)
                        
                        # 计算CPU和内存使用率（假设请求资源占比）
                        cpu_usage = min(1.0, cpu_request / cpu_capacity if cpu_capacity > 0 else 1.0)
                        memory_usage = min(1.0, memory_request / memory_capacity if memory_capacity > 0 else 1.0)
                        
                        resource_info[node.name] = {
                            'cpu': cpu_usage,
                            'memory': memory_usage
                        }
                    except Exception as e:
                        logger.warning(f"处理节点 {node.name} 的资源信息时出错: {str(e)}")
                        resource_info[node.name] = {
                            'cpu': 0.8,  # 默认高负载
                            'memory': 0.8  # 默认高负载
                        }
                
                # 使用优化器服务计算节点得分
                priority_response = self.optimizer_service.calculate_node_scores(
                    pod_name=request.pod.name,
                    nodes=node_names,
                    resource_info=resource_info
                )
                
                # 转换为需要的格式
                node_scores = [
                    NodeScore(name=priority["Host"], score=priority["Score"])
                    for priority in priority_response.hostPriorityList
                ]
                
                response.nodes = node_scores
                response.node_names = [node_score.name for node_score in node_scores]
            else:
                # 如果不使用优化器，创建默认得分的节点列表
                node_scores = [NodeScore(name=node.name, score=50) for node in filtered_nodes]
                response.nodes = node_scores
                response.node_names = [node.name for node in filtered_nodes]

            response.failed_nodes = failed_nodes
            return response

        # 没有特殊要求，检查所有节点
        logger.info(f"检查所有节点的资源情况")
        filtered_nodes = []
        for node in request.nodes:
            is_fit, reason = await check_node_resources(
                node,
                cpu_request,
                memory_request
            )
            if is_fit:
                filtered_nodes.append(node)
            else:
                failed_nodes[node.name] = reason

        # 使用优化器对节点进行排序
        if self.enable_optimizer and filtered_nodes:
            # 准备节点名称和资源信息
            node_names = [node.name for node in filtered_nodes]
            resource_info = {}
            
            # 构建资源信息字典
            for node in filtered_nodes:
                try:
                    allocatable = node.status.allocatable
                    cpu_capacity = parse_resource_value(allocatable.cpu)
                    memory_capacity = parse_resource_value(allocatable.memory)
                    
                    # 计算CPU和内存使用率（假设请求资源占比）
                    cpu_usage = min(1.0, cpu_request / cpu_capacity if cpu_capacity > 0 else 1.0)
                    memory_usage = min(1.0, memory_request / memory_capacity if memory_capacity > 0 else 1.0)
                    
                    resource_info[node.name] = {
                        'cpu': cpu_usage,
                        'memory': memory_usage
                    }
                except Exception as e:
                    logger.warning(f"处理节点 {node.name} 的资源信息时出错: {str(e)}")
                    resource_info[node.name] = {
                        'cpu': 0.8,  # 默认高负载
                        'memory': 0.8  # 默认高负载
                    }
            
            # 使用优化器服务计算节点得分
            priority_response = self.optimizer_service.calculate_node_scores(
                pod_name=request.pod.name,
                nodes=node_names,
                resource_info=resource_info
            )
            
            # 转换为需要的格式
            node_scores = [
                NodeScore(name=priority["Host"], score=priority["Score"])
                for priority in priority_response.hostPriorityList
            ]
            
            response.nodes = node_scores
            response.node_names = [node_score.name for node_score in node_scores]
        else:
            # 如果不使用优化器，创建默认得分的节点列表
            node_scores = [NodeScore(name=node.name, score=50) for node in filtered_nodes]
            response.nodes = node_scores
            response.node_names = [node.name for node in filtered_nodes]

        response.failed_nodes = failed_nodes
        return response

    def create_pod(self, namespace: str, pod: Dict) -> Dict:
        """创建Pod
        
        Args:
            namespace: 命名空间
            pod: Pod定义
            
        Returns:
            Dict: 创建的Pod信息
        """
        return self.k8s_service.create_pod(namespace, pod)

    def delete_pod(self, namespace: str, pod_name: str):
        """删除Pod
        
        Args:
            namespace: 命名空间
            pod_name: Pod名称
        """
        self.k8s_service.delete_pod(namespace, pod_name)

    def get_pod_status(self, namespace: str, pod_name: str) -> Dict:
        """获取Pod状态
        
        Args:
            namespace: 命名空间
            pod_name: Pod名称
            
        Returns:
            Dict: Pod状态信息
        """
        return self.k8s_service.get_pod_status(namespace, pod_name)

    async def select_node_for_pod(self, pod_name: str) -> Tuple[Optional[str], Optional[str]]:
        """为Pod选择最佳节点
        
        根据之前计算的优先级方案选择最佳节点
        
        Args:
            pod_name: Pod名称
            
        Returns:
            Tuple[Optional[str], Optional[str]]: (节点名称, 错误信息)，如果没有可用节点或出错，则节点名称为None
        """
        try:
            # 获取Pod的调度方案
            plan = self._plan_store.get_plan(pod_name)
            if not plan:
                return None, f"未找到Pod {pod_name} 的调度方案"
                
            # 从方案中获取选定的节点
            selected_node = plan.get('selected_node')
            if not selected_node:
                return None, f"Pod {pod_name} 的调度方案中没有选定节点"
                
            logger.info(f"为Pod {pod_name} 选择节点 {selected_node}")
            return selected_node, None
            
        except Exception as e:
            error_msg = f"为Pod {pod_name} 选择节点时出错: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    async def get_node_scores(self, pod_name: str) -> List[NodeScore]:
        """获取节点得分列表
        
        Args:
            pod_name: Pod名称
            
        Returns:
            List[NodeScore]: 节点得分列表
        """
        try:
            # 获取Pod的调度方案
            plan = self._plan_store.get_plan(pod_name)
            if not plan:
                logger.warning(f"未找到Pod {pod_name} 的调度方案")
                return []
                
            # 从方案中获取节点得分
            node_scores = plan.get('node_scores', {})
            
            # 转换为NodeScore对象列表
            result = [
                NodeScore(name=node, score=int(float(score) * 100))
                for node, score in node_scores.items()
            ]
            
            # 按得分降序排序
            result.sort(key=lambda x: x.score, reverse=True)
            
            return result
            
        except Exception as e:
            logger.error(f"获取Pod {pod_name} 的节点得分时出错: {str(e)}")
            return []
            
    def convert_host_priorities_to_node_scores(
        self, 
        host_priorities: List[HostPriority]
    ) -> List[NodeScore]:
        """将HostPriority列表转换为NodeScore列表
        
        Args:
            host_priorities: HostPriority列表
            
        Returns:
            List[NodeScore]: NodeScore列表
        """
        return [
            NodeScore(
                name=hp.host,
                score=int(float(hp.score) * 100)
            )
            for hp in host_priorities
        ]
