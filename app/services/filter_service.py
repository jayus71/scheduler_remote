"""节点过滤服务"""
from typing import Dict, Tuple, List
from loguru import logger
import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, LpBinary, lpSum

from app.schemas.filter import FilterRequest, FilterResult, NodeScore
from app.schemas.computing_power import ComputingPowerFilterResult
from app.services.computing_power_service import ComputingPowerService
from app.services.k8s_service import KubernetesService
from app.core.config import settings
from app.utils.resource_parser import parse_resource_value


class FilterService:
    """节点过滤服务类"""

    def __init__(self):
        """初始化过滤服务"""
        self.k8s_service = KubernetesService()
        self.computing_power_service = ComputingPowerService()
        # 从settings获取资源权重配置
        self.resource_weights = {
            "cpu": settings.WEIGHT_CPU,
            "memory": settings.WEIGHT_MEMORY
        }
        # 算力权重配置
        self.computing_power_weights = {
            "cpu_power": settings.WEIGHT_CPU_POWER,
            "gpu_power": settings.WEIGHT_GPU_POWER,
            "fpga_power": settings.WEIGHT_FPGA_POWER
        }
        # 是否启用算力量化筛选
        self.enable_computing_power_filter = settings.ENABLE_COMPUTING_POWER_FILTER
        # 是否启用整数规划算法进行预过滤
        self.enable_integer_programming = True
        logger.info("过滤服务初始化完成，已启用整数规划算法进行节点预过滤")

    async def filter_nodes(self, request: FilterRequest) -> FilterResult:
        """节点预过滤逻辑
        
        Args:
            request: 过滤请求
            
        Returns:
            FilterResult: 过滤响应
        """
        filter_result = FilterResult()
        failed_nodes: Dict[str, str] = {}

        # 如果没有提供节点列表，从Kubernetes获取
        if not request.nodes:
            request.nodes = self.k8s_service.list_nodes()
            logger.info(f"从Kubernetes获取到 {len(request.nodes)} 个节点")

        # 获取所有容器的资源请求
        total_cpu_request = 0
        total_memory_request = 0
        total_gpu_request = 0
        
        for container in request.pod.spec.containers:
            try:
                # 尝试获取资源请求，如果不存在则使用默认值
                if container.resources and container.resources.requests:
                    cpu_request = parse_resource_value(container.resources.requests.cpu)
                    memory_request = parse_resource_value(container.resources.requests.memory)
                    # 尝试获取GPU请求
                    gpu_request = 0
                    if hasattr(container.resources.requests, 'nvidia.com/gpu'):
                        try:
                            gpu_request = int(container.resources.requests.gpu)
                        except (ValueError, TypeError):
                            logger.warning(f"无法解析容器 {container.name} 的GPU请求，使用默认值0")
                            gpu_request = 0
                else:
                    # 如果没有资源请求，使用默认值
                    logger.warning(f"容器 {container.name} 没有资源请求，使用默认值")
                    cpu_request = parse_resource_value("100m")
                    memory_request = parse_resource_value("128Mi")
                    gpu_request = 0
            except Exception as e:
                # 如果解析失败，使用默认值
                logger.warning(f"解析容器 {container.name} 的资源请求失败: {str(e)}，使用默认值")
                cpu_request = parse_resource_value("100m")
                memory_request = parse_resource_value("128Mi")
                gpu_request = 0
            
            total_cpu_request += cpu_request
            total_memory_request += memory_request
            total_gpu_request += gpu_request
            logger.debug(f"容器 {container.name} 资源需求: CPU={cpu_request}m, 内存={memory_request}MB, GPU={gpu_request}")

        logger.info(f"Pod总资源需求: CPU={total_cpu_request}m, 内存={total_memory_request}MB, GPU={total_gpu_request}")

        # 如果指定了具体节点
        if request.pod.spec.nodeName:
            target_node = next(
                (node for node in request.nodes if node.name == request.pod.spec.nodeName),
                None
            )
            if not target_node:
                filter_result.error = f"指定的节点 {request.pod.spec.nodeName} 不存在"
                return filter_result

            is_fit, reason = await self._check_node_resources(
                target_node,
                total_cpu_request,
                total_memory_request,
                total_gpu_request
            )
            if is_fit:
                # 如果启用了算力量化筛选，还需要检查算力需求
                if self.enable_computing_power_filter:
                    # 获取量化后的算力需求
                    quantified_power = await self.computing_power_service.quantify_computing_power(request.pod.spec.containers)
                    if quantified_power:
                        # 合并所有容器的算力需求
                        total_cpu_power = sum(power[0] for power in quantified_power)
                        total_gpu_power = sum(power[1] for power in quantified_power)
                        total_fpga_power = sum(power[2] for power in quantified_power)
                        total_power = (total_cpu_power, total_gpu_power, total_fpga_power)
                        
                        # 检查算力是否满足需求
                        computing_power_result = await self._filter_nodes_by_computing_power(
                            [target_node],
                            total_power
                        )
                        if target_node.name not in computing_power_result.suitable_nodes:
                            failed_nodes[target_node.name] = computing_power_result.unsuitable_nodes[target_node.name]
                            filter_result.failed_nodes = failed_nodes
                            return filter_result

                node_score = NodeScore(name=target_node.name, score=100)  # 指定节点默认得分为100
                filter_result.nodes = [node_score]
                filter_result.node_names = [target_node.name]
            else:
                failed_nodes[target_node.name] = reason
            filter_result.failed_nodes = failed_nodes
            return filter_result

        # 如果指定了节点选择器
        if request.pod.spec.node_selector:
            logger.info(f"使用节点选择器: {request.pod.spec.node_selector}")
            filtered_nodes = []
            for node in request.nodes:
                # 检查节点是否匹配所有选择器
                if all(node.labels.get(key) == value for key, value in request.pod.spec.node_selector.items()):
                    filtered_nodes.append(node)
                else:
                    failed_nodes[node.name] = "节点标签不匹配"
            request.nodes = filtered_nodes
            logger.info(f"节点选择器过滤后剩余 {len(request.nodes)} 个节点")

        # 向算力路由发送业务算力量化请求
        total_cpu_power = 0.0
        total_gpu_power = 0.0
        total_fpga_power = 0.0
        total_power = (total_cpu_power, total_gpu_power, total_fpga_power)
        
        if self.enable_computing_power_filter:
            logger.info("向算力路由发送业务算力量化请求")
            try:
                quantified_power = await self.computing_power_service.quantify_computing_power(request.pod.spec.containers)
                
                if quantified_power:
                    # 合并所有容器的算力需求
                    total_cpu_power = sum(power[0] for power in quantified_power)
                    total_gpu_power = sum(power[1] for power in quantified_power)
                    total_fpga_power = sum(power[2] for power in quantified_power)
                    total_power = (total_cpu_power, total_gpu_power, total_fpga_power)
                    logger.info(f"业务算力量化结果: CPU算力={total_cpu_power}, GPU算力={total_gpu_power}, FPGA算力={total_fpga_power}")
            except Exception as e:
                logger.error(f"获取算力量化结果失败: {str(e)}")
                logger.warning("将使用默认算力需求进行整数规划")

        # 使用整数规划算法筛选节点
        if self.enable_integer_programming:
            logger.info("使用整数规划算法筛选节点")
            try:
                # 收集所有节点资源信息和算力信息
                node_resource_info = []
                node_computing_power_info = []
                
                for node in request.nodes:
                    # 获取节点资源
                    cpu_capacity = parse_resource_value(str(node.capacity.cpu))
                    memory_capacity = parse_resource_value(str(node.capacity.memory))
                    
                    # 获取节点GPU资源
                    gpu_capacity = 0
                    if hasattr(node.capacity, 'nvidia.com/gpu'):
                        try:
                            gpu_capacity = int(node.capacity.gpu)
                        except (ValueError, TypeError):
                            logger.warning(f"无法解析节点 {node.name} 的GPU容量，使用默认值0")
                    
                    # 收集节点资源信息
                    node_resource_info.append({
                        'name': node.name,
                        'cpu': cpu_capacity,
                        'memory': memory_capacity,
                        'gpu': gpu_capacity
                    })
                    
                    # 计算CPU和内存使用率
                    cpu_usage = min(1.0, total_cpu_request / cpu_capacity if cpu_capacity > 0 else 1.0)
                    memory_usage = min(1.0, total_memory_request / memory_capacity if memory_capacity > 0 else 1.0)
                    
                    # 估算节点算力
                    cpu_cores = cpu_capacity / 1000  # 转换为核心数
                    cpu_power = cpu_cores * 2  # 假设平均主频为2GHz
                    
                    gpu_power = 0.0
                    if gpu_capacity > 0:
                        gpu_power = gpu_capacity * 10.0  # 假设每个GPU提供10 TFLOPS
                    
                    # 收集节点算力信息
                    node_computing_power_info.append({
                        'name': node.name,
                        'cpu_power': cpu_power,
                        'gpu_power': gpu_power,
                        'cpu_usage': cpu_usage,
                        'memory_usage': memory_usage,
                        'gpu_available': gpu_capacity > 0
                    })
                
                # 使用整数规划选择节点，将算力量化结果传递给整数规划算法
                suitable_nodes = self._integer_programming_filter(
                    node_resource_info,
                    node_computing_power_info,
                    total_cpu_request,
                    total_memory_request,
                    total_gpu_request,
                    True,  # 始终检查算力需求
                    total_power  # 传递算力量化结果
                )
                
                # 更新候选节点和失败节点
                candidate_nodes = [node for node in request.nodes if node.name in suitable_nodes]
                for node in request.nodes:
                    if node.name not in suitable_nodes:
                        failed_nodes[node.name] = "通过整数规划算法筛选未选中"
                
                logger.info(f"整数规划筛选后剩余 {len(candidate_nodes)} 个候选节点")
            except Exception as e:
                logger.error(f"整数规划筛选节点时出错: {str(e)}", exc_info=True)
                # 如果整数规划失败，回退到传统方法
                logger.warning("整数规划筛选失败，回退到传统资源检查方法")
                candidate_nodes = []
                for node in request.nodes:
                    is_fit, reason = await self._check_node_resources(
                        node,
                        total_cpu_request,
                        total_memory_request,
                        total_gpu_request
                    )
                    if is_fit:
                        candidate_nodes.append(node)
                    else:
                        failed_nodes[node.name] = reason
        else:
            # 使用传统方法检查节点资源
            candidate_nodes = []
            for node in request.nodes:
                is_fit, reason = await self._check_node_resources(
                    node,
                    total_cpu_request,
                    total_memory_request,
                    total_gpu_request
                )
                if is_fit:
                    candidate_nodes.append(node)
                else:
                    failed_nodes[node.name] = reason

        logger.info(f"资源检查后剩余 {len(candidate_nodes)} 个候选节点")
        
        # 如果启用了算力量化筛选，进行算力需求分析和筛选
        if self.enable_computing_power_filter and candidate_nodes and not self.enable_integer_programming:
            # 只有在没有使用整数规划时才单独进行算力筛选，因为整数规划已经考虑了算力因素
            logger.info("使用算力量化进行筛选")
            # 获取量化后的算力需求
            quantified_power = await self.computing_power_service.quantify_computing_power(request.pod.spec.containers)
            
            if quantified_power:
                # 合并所有容器的算力需求
                total_cpu_power = sum(power[0] for power in quantified_power)
                total_gpu_power = sum(power[1] for power in quantified_power)
                total_fpga_power = sum(power[2] for power in quantified_power)
                total_power = (total_cpu_power, total_gpu_power, total_fpga_power)
                
                # 过滤满足算力需求的节点
                computing_power_result = await self._filter_nodes_by_computing_power(
                    candidate_nodes, 
                    total_power
                )
                
                # 更新候选节点列表
                candidate_node_names = computing_power_result.suitable_nodes
                candidate_nodes = [node for node in candidate_nodes if node.name in candidate_node_names]
                
                # 更新失败节点列表
                for node_name, reason in computing_power_result.unsuitable_nodes.items():
                    failed_nodes[node_name] = reason
                
                logger.info(f"算力筛选后剩余 {len(candidate_nodes)} 个候选节点")

        # 根据整数规划筛选结果直接返回节点得分，不使用优化器（遗传算法）
        logger.info("使用整数规划结果生成节点列表")
        
        # 填充节点内部IP和主机名字段
        for node in candidate_nodes:
            # 从addresses中提取信息
            if node.addresses:
                node.internalIP = node.addresses.internalIP
                node.hostname = node.addresses.hostname
            # 如果addresses不存在，尝试从其他地方获取信息
            else:
                # 尝试获取内部IP
                if hasattr(node, 'status') and hasattr(node.status, 'addresses'):
                    for address in node.status.addresses:
                        if address.type == "InternalIP":
                            node.internalIP = address.address
                        elif address.type == "Hostname":
                            node.hostname = address.address
        
        # 保存节点信息和节点名称
        filter_result.nodes = candidate_nodes  # 保存完整的节点信息
        filter_result.node_names = [node.name for node in candidate_nodes]
        filter_result.failed_nodes = failed_nodes
        
        return filter_result
        
    def _integer_programming_filter(
        self, 
        node_resource_info: List[Dict], 
        node_computing_power_info: List[Dict],
        cpu_request: float,
        memory_request: float,
        gpu_request: int,
        check_computing_power: bool = False,
        required_power: Tuple[int, float, float] = (0, 0, 0)
    ) -> List[str]:
        """使用整数规划算法选择最优节点集合
        
        Args:
            node_resource_info: 节点资源信息列表
            node_computing_power_info: 节点算力信息列表
            cpu_request: CPU请求量
            memory_request: 内存请求量
            gpu_request: GPU请求量
            check_computing_power: 是否检查算力需求
            required_power: 所需算力元组 (cpu_power, gpu_power, fpga_power)
            
        Returns:
            List[str]: 满足条件的节点名称列表
        """
        try:
            # 创建整数规划问题（最大化选中节点数量）
            prob = LpProblem("NodeSelection", LpMaximize)
            
            # 创建决策变量（每个节点是否被选中）
            x = {}
            for i, node_info in enumerate(node_resource_info):
                x[i] = LpVariable(f"x_{i}", 0, 1, LpBinary)
            
            # 设置目标函数：最大化满足条件的节点数量
            prob += lpSum(x.values()), "Maximize the number of suitable nodes"
            
            # 添加资源约束
            for i, node_info in enumerate(node_resource_info):
                # 只有当节点被选中时约束才生效
                # CPU约束
                prob += node_info['cpu'] * x[i] >= cpu_request * x[i], f"CPU constraint for node {i}"
                
                # 内存约束
                prob += node_info['memory'] * x[i] >= memory_request * x[i], f"Memory constraint for node {i}"
                
                # GPU约束（如果需要）
                if gpu_request > 0:
                    prob += node_info['gpu'] * x[i] >= gpu_request * x[i], f"GPU constraint for node {i}"
            
            # 添加算力约束
            if check_computing_power:
                # 解析所需算力
                required_cpu_power, required_gpu_power, required_fpga_power = required_power
                
                # 如果算力量化结果为0，使用启发式方法估算算力需求
                if required_cpu_power <= 0:
                    required_cpu_power = cpu_request / 1000 * 2  # 假设每核心2GHz
                
                if required_gpu_power <= 0 and gpu_request > 0:
                    required_gpu_power = gpu_request * 5.0  # 假设每GPU需要5TFLOPS
                
                logger.info(f"整数规划算力约束: CPU算力={required_cpu_power}, GPU算力={required_gpu_power}, FPGA算力={required_fpga_power}")
                
                for i, node_power in enumerate(node_computing_power_info):
                    # CPU算力约束
                    if required_cpu_power > 0:
                        prob += node_power['cpu_power'] * x[i] >= required_cpu_power * x[i], f"CPU power constraint for node {i}"
                    
                    # GPU算力约束（如果需要）
                    if required_gpu_power > 0:
                        prob += node_power['gpu_power'] * x[i] >= required_gpu_power * x[i], f"GPU power constraint for node {i}"
            
            # 解决问题
            prob.solve()
            
            # 获取选中的节点
            selected_nodes = []
            for i, node_info in enumerate(node_resource_info):
                if x[i].value() == 1:
                    selected_nodes.append(node_info['name'])
            
            logger.info(f"整数规划选中了 {len(selected_nodes)} 个节点")
            return selected_nodes
        except Exception as e:
            logger.error(f"执行整数规划算法时出错: {str(e)}")
            # 如果整数规划失败，返回所有节点，由后续过程按照传统方法过滤
            return [node_info['name'] for node_info in node_resource_info]

    async def _check_node_resources(self, node, cpu_request, memory_request, gpu_request=0) -> Tuple[bool, str]:
        """检查节点资源是否满足需求
        
        Args:
            node: 节点信息
            cpu_request: CPU请求量（毫核）
            memory_request: 内存请求量（MB）
            gpu_request: GPU请求量
            
        Returns:
            Tuple[bool, str]: (是否满足, 不满足原因)
        """
        # 获取节点的CPU和内存容量
        node_cpu_capacity = parse_resource_value(str(node.capacity.cpu))
        node_memory_capacity = parse_resource_value(str(node.capacity.memory))
        
        # 获取节点的GPU容量
        node_gpu_capacity = 0
        if hasattr(node.capacity, 'nvidia.com/gpu'):
            try:
                node_gpu_capacity = int(node.capacity.gpu)
            except (ValueError, TypeError):
                logger.warning(f"无法解析节点 {node.name} 的GPU容量，使用默认值0")
                node_gpu_capacity = 0

        # 计算可用资源
        available_cpu = node_cpu_capacity
        available_memory = node_memory_capacity
        available_gpu = node_gpu_capacity

        # 检查CPU资源
        if available_cpu < cpu_request:
            return False, f"CPU资源不足: 需要{cpu_request}m, 可用{available_cpu}m"

        # 检查内存资源
        if available_memory < memory_request:
            return False, f"内存资源不足: 需要{memory_request}MB, 可用{available_memory}MB"
            
        # 检查GPU资源
        if gpu_request > 0 and available_gpu < gpu_request:
            return False, f"GPU资源不足: 需要{gpu_request}, 可用{available_gpu}"

        return True, "资源充足"
        
    async def _filter_nodes_by_computing_power(self, nodes: List, required_power: Tuple[int, float, float]) -> ComputingPowerFilterResult:
        """
        根据算力需求筛选节点
        
        Args:
            nodes: 候选节点列表
            required_power: 所需算力元组 (cpu_power, gpu_power, fpga_power)
            
        Returns:
            ComputingPowerFilterResult: 筛选结果
        """
        result = ComputingPowerFilterResult()
        result.suitable_nodes = []
        result.unsuitable_nodes = {}
        
        # 解析所需算力
        required_cpu_power, required_gpu_power, required_fpga_power = required_power
        
        for node in nodes:
            try:
                # 估算节点的算力容量
                node_cpu_cores = parse_resource_value(str(node.capacity.cpu)) / 1000  # 转换为核心数
                node_cpu_power = node_cpu_cores * 2  # 假设平均主频为2GHz
                
                # 从节点容量和标签中获取GPU和FPGA能力
                node_gpu_power = 0.0
                node_fpga_power = 0.0
                
                # 检查GPU容量
                if hasattr(node.capacity, 'nvidia.com/gpu'):
                    try:
                        gpu_count = int(node.capacity.gpu)
                        # 简单假设每个GPU提供10 TFLOPS
                        node_gpu_power = gpu_count * 10.0
                    except (ValueError, TypeError):
                        logger.warning(f"无法解析节点 {node.name} 的GPU容量，使用默认值0")
                
                # 检查FPGA标签
                if node.labels.get("fpga") == "true" or node.labels.get("fpga.intel.com/arria10") == "true":
                    # 从标签中推断FPGA数量
                    fpga_count = 1  # 默认1个
                    if "fpga.intel.com/arria10.count" in node.labels:
                        try:
                            fpga_count = int(node.labels["fpga.intel.com/arria10.count"])
                        except ValueError:
                            fpga_count = 1
                    
                    # 简单假设每个FPGA提供5 TFLOPS
                    node_fpga_power = fpga_count * 5.0
                
                # 检查算力是否满足需求
                if node_cpu_power < required_cpu_power:
                    result.unsuitable_nodes[node.name] = f"CPU算力不足: 需要{required_cpu_power}GHz, 可用{node_cpu_power}GHz"
                    continue
                    
                if required_gpu_power > 0 and node_gpu_power < required_gpu_power:
                    result.unsuitable_nodes[node.name] = f"GPU算力不足: 需要{required_gpu_power}TFLOPS, 可用{node_gpu_power}TFLOPS"
                    continue
                    
                if required_fpga_power > 0 and node_fpga_power < required_fpga_power:
                    result.unsuitable_nodes[node.name] = f"FPGA算力不足: 需要{required_fpga_power}TFLOPS, 可用{node_fpga_power}TFLOPS"
                    continue
                
                # 如果通过所有检查，添加到结果中
                result.suitable_nodes.append(node.name)
                
            except Exception as e:
                logger.error(f"处理节点 {node.name} 的算力信息时出错: {str(e)}")
                result.unsuitable_nodes[node.name] = f"处理算力信息时出错: {str(e)}"
                
        return result
