"""优化器模块"""
import random
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger
import math

from app.schemas.common import Node, NodeScore
from app.core.config import settings
from app.schemas.priority import HostPriority, PriorityResponse, RLState, RLAction, RLExperience
from app.services.rl_agent import RLAgent


class Ant:
    """蚂蚁类，用于蚁群算法"""
    
    def __init__(self, nodes: List[str]):
        """初始化蚂蚁
        
        Args:
            nodes: 节点列表
        """
        self.nodes = nodes
        self.path = []         # 蚂蚁的路径
        self.fitness = 0.0     # 蚂蚁路径的适应度
        self.weights = []      # 节点权重


class OptimizerService:
    """优化器服务，负责节点优先级计算"""
    
    def __init__(self, plan_store=None):
        """初始化优化器服务
        
        Args:
            plan_store: 可选的调度方案存储服务实例
        """
        # 保存调度方案存储实例
        self.plan_store = plan_store
        
        # 评分权重
        self.weights = {
            'basic_resources': 0.4,  # 基础资源权重
            'computing_power': 0.3,  # 算力权重
            'network': 0.3,          # 网络权重
        }
        
        # 从配置文件加载遗传算法参数
        self.ga_params = {
            'population_size': settings.GA_POPULATION_SIZE,        # 种群大小
            'generations': settings.GA_GENERATIONS,                # 迭代代数
            'mutation_rate': settings.GA_MUTATION_RATE,            # 变异率
            'elitism_rate': 0.1,                                   # 精英保留率
            'weight_resource_utilization': settings.GA_WEIGHT_CPU,  # 资源利用率权重（原weight_cpu）
            'weight_resource_balance': settings.GA_WEIGHT_MEMORY,   # 资源均衡性权重（原weight_memory）
            'weight_computing': settings.GA_WEIGHT_BALANCE,        # 算力权重（原weight_balance）
            'weight_network': getattr(settings, 'GA_WEIGHT_NETWORK', 0.25),  # 网络权重
        }
        
        # 粒子群优化(PSO)参数
        self.pso_params = {
            'swarm_size': getattr(settings, 'PSO_SWARM_SIZE', 30),        # 粒子群大小
            'max_iterations': getattr(settings, 'PSO_MAX_ITERATIONS', 50),  # 最大迭代次数
            'inertia_weight': getattr(settings, 'PSO_INERTIA_WEIGHT', 0.7),  # 惯性权重
            'cognitive_coef': getattr(settings, 'PSO_COGNITIVE_COEF', 1.5),  # 认知系数
            'social_coef': getattr(settings, 'PSO_SOCIAL_COEF', 1.5),      # 社会系数
            'weight_resource_utilization': getattr(settings, 'PSO_WEIGHT_CPU', 0.25),  # 资源利用率权重（原weight_cpu）
            'weight_resource_balance': getattr(settings, 'PSO_WEIGHT_MEMORY', 0.25),  # 资源均衡性权重（原weight_memory）
            'weight_computing': getattr(settings, 'PSO_WEIGHT_BALANCE', 0.25),  # 算力权重（原weight_balance）
            'weight_network': getattr(settings, 'PSO_WEIGHT_NETWORK', 0.25),  # 网络权重
        }
        
        # 模拟退火算法(SA)参数
        self.sa_params = {
            'initial_temp': getattr(settings, 'SA_INITIAL_TEMP', 100.0),    # 初始温度
            'cooling_rate': getattr(settings, 'SA_COOLING_RATE', 0.95),     # 冷却速率
            'min_temp': getattr(settings, 'SA_MIN_TEMP', 0.1),              # 最小温度
            'iterations_per_temp': getattr(settings, 'SA_ITERATIONS_PER_TEMP', 10),  # 每个温度下的迭代次数
            'weight_resource_utilization': getattr(settings, 'SA_WEIGHT_CPU', 0.25),  # 资源利用率权重（原weight_cpu）
            'weight_resource_balance': getattr(settings, 'SA_WEIGHT_MEMORY', 0.25),  # 资源均衡性权重（原weight_memory）
            'weight_computing': getattr(settings, 'SA_WEIGHT_BALANCE', 0.25),  # 算力权重（原weight_balance）
            'weight_network': getattr(settings, 'SA_WEIGHT_NETWORK', 0.25), # 网络权重
        }
        
        # 蚁群算法(ACO)参数
        self.aco_params = {
            'ants_count': getattr(settings, 'ACO_ANTS_COUNT', 20),          # 蚂蚁数量
            'iterations': getattr(settings, 'ACO_ITERATIONS', 50),          # 迭代次数
            'alpha': getattr(settings, 'ACO_ALPHA', 1.0),                   # 信息素重要程度系数
            'beta': getattr(settings, 'ACO_BETA', 2.0),                     # 启发函数重要程度系数
            'rho': getattr(settings, 'ACO_RHO', 0.5),                       # 信息素挥发系数
            'q': getattr(settings, 'ACO_Q', 100.0),                         # 信息素增加强度系数
            'weight_resource_utilization': getattr(settings, 'ACO_WEIGHT_CPU', 0.25),  # 资源利用率权重（原weight_cpu）
            'weight_resource_balance': getattr(settings, 'ACO_WEIGHT_MEMORY', 0.25),  # 资源均衡性权重（原weight_memory）
            'weight_computing': getattr(settings, 'ACO_WEIGHT_BALANCE', 0.25),  # 算力权重（原weight_balance）
            'weight_network': getattr(settings, 'ACO_WEIGHT_NETWORK', 0.25),  # 网络权重
        }
        
        # 优化方法选择
        self.optimization_method = getattr(settings, 'OPTIMIZATION_METHOD', 'genetic')  # 默认使用遗传算法
        
        # 强化学习配置
        self.enable_rl = settings.ENABLE_RL
        self.rl_hybridization_weight = settings.RL_HYBRIDIZATION_WEIGHT  # 强化学习权重
        
        # 初始化强化学习代理
        self.rl_agent = None
        if self.enable_rl:
            try:
                self.rl_agent = RLAgent()
                logger.info("强化学习代理初始化完成")
            except Exception as e:
                logger.error(f"初始化强化学习代理时出错: {str(e)}")
                self.enable_rl = False
        
        # 上一次调度的状态和动作
        self.last_state = None
        self.last_action = None
        self.last_node_scores = None
        self.last_selected_node = None
        
        logger.info("优化器服务初始化完成，已加载优化算法参数: "
                   f"优化方法={self.optimization_method}, "
                   f"遗传算法种群大小={self.ga_params['population_size']}, "
                   f"PSO粒子群大小={self.pso_params['swarm_size']}, "
                   f"SA初始温度={self.sa_params['initial_temp']}, "
                   f"ACO蚂蚁数量={self.aco_params['ants_count']}, "
                   f"强化学习启用={self.enable_rl}")
        
    def calculate_node_scores(
        self,
        pod_name: str,
        nodes: List[str],
        resource_info: Dict[str, Dict[str, float]],
        computing_power_info: Optional[Dict[str, Dict[str, Any]]] = None,
        network_info: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> PriorityResponse:
        """计算节点得分
        
        Args:
            pod_name: Pod名称或Pod对象
            nodes: 节点列表
            resource_info: 资源信息 {node: {'cpu': usage, 'memory': usage}}
            computing_power_info: 算力信息 {node: {'cpu_load': load, 'gpu_load': load}}
            network_info: 网络信息 {node: {'latency': ms, 'bandwidth': mbps}}
            
        Returns:
            PriorityResponse: 节点优先级响应，符合接口标识符Schedule-priority-response
        """
        try:
            # 确保pod_name是字符串类型
            pod_identifier = str(pod_name) if pod_name else "unknown-pod"
            
            # 确保nodes不为空
            if not nodes:
                logger.warning(f"Pod {pod_identifier}: 节点列表为空，无法计算得分")
                return PriorityResponse(hostPriorityList=[], error="节点列表为空")
                
            # 确保resource_info不为空且格式正确
            if not resource_info:
                logger.warning(f"Pod {pod_identifier}: 资源信息为空，使用默认资源信息")
                resource_info = {node: {'cpu': 0.5, 'memory': 0.5} for node in nodes}
            
            # 节点特征矩阵: 每个节点对应一个特征向量[cpu使用率, 内存使用率, 算力得分, 网络得分]
            # 计算基础数据
            resource_scores = self._calculate_resource_scores(nodes, resource_info)
            computing_scores = self._calculate_computing_scores(nodes, computing_power_info)
            network_scores = self._calculate_network_scores(nodes, network_info)
            
            node_features = {}
            for node in nodes:
                cpu_usage = resource_info.get(node, {}).get('cpu', 1.0)  # 默认为1.0表示资源已满
                memory_usage = resource_info.get(node, {}).get('memory', 1.0)
                
                node_features[node] = [
                    cpu_usage,  # CPU使用率，越低越好
                    memory_usage,  # 内存使用率，越低越好
                    computing_scores.get(node, 0),  # 算力得分，越高越好
                    network_scores.get(node, 0)  # 网络得分，越高越好
                ]
            
            # 更新上一个状态的奖励并存储经验（如果有）
            if self.enable_rl and self.rl_agent and self.last_state and self.last_action and self.last_selected_node:
                self._update_last_experience(resource_info)
            
            # 根据选择的优化方法计算节点得分
            if self.optimization_method == 'pso':
                # 使用粒子群优化
                algorithm_scores = self._particle_swarm_optimization(
                    nodes, 
                    resource_info, 
                    computing_power_info, 
                    network_info,
                    resource_scores=resource_scores,
                    computing_scores=computing_scores,
                    network_scores=network_scores
                )
                logger.info(f"Pod {pod_identifier}: 使用粒子群优化计算节点得分")
            elif self.optimization_method == 'sa':
                # 使用模拟退火算法
                algorithm_scores = self._simulated_annealing(
                    nodes, 
                    resource_info, 
                    computing_power_info, 
                    network_info,
                    resource_scores=resource_scores,
                    computing_scores=computing_scores,
                    network_scores=network_scores
                )
                logger.info(f"Pod {pod_identifier}: 使用模拟退火算法计算节点得分")
            elif self.optimization_method == 'aco':
                # 使用蚁群算法
                algorithm_scores = self._ant_colony_optimization(
                    nodes, 
                    resource_info, 
                    computing_power_info, 
                    network_info,
                    resource_scores=resource_scores,
                    computing_scores=computing_scores,
                    network_scores=network_scores
                )
                logger.info(f"Pod {pod_identifier}: 使用蚁群算法计算节点得分")
            else:
                # 使用遗传算法
                algorithm_scores = self._genetic_algorithm(
                    nodes, 
                    resource_info, 
                    computing_power_info, 
                    network_info,
                    resource_scores=resource_scores,
                    computing_scores=computing_scores,
                    network_scores=network_scores
                )
                logger.info(f"Pod {pod_identifier}: 使用遗传算法计算节点得分")
            
            final_scores = {}
            
            if self.enable_rl and self.rl_agent:
                try:
                    # 创建当前状态
                    current_state = self._create_rl_state(pod_identifier, node_features, resource_info)
                    
                    # 使用强化学习代理选择动作
                    rl_node_weights = self.rl_agent.select_action(current_state, nodes)
                    
                    # 记录当前状态和动作，用于下一次计算奖励
                    self.last_state = current_state
                    self.last_action = RLAction(node_weights=rl_node_weights)
                    self.last_node_scores = {node: algorithm_scores.get(node, 0.0) for node in nodes}
                    
                    # 混合强化学习和算法的结果
                    rl_weight = self.rl_hybridization_weight
                    algorithm_weight = 1.0 - rl_weight
                    
                    for node in nodes:
                        rl_score = rl_node_weights.get(node, 0.0)
                        algo_score = algorithm_scores.get(node, 0.0)
                        
                        # 加权平均，确保考虑所有资源因素
                        # 资源、算力和网络的综合评分
                        resource_score = resource_scores.get(node, 0.0)
                        computing_score = computing_scores.get(node, 0.0)
                        network_score = network_scores.get(node, 0.0)
                        
                        # 确保RL得分也考虑了所有资源因素
                        resource_factor = 0.4 * resource_score
                        computing_factor = 0.3 * computing_score
                        network_factor = 0.3 * network_score
                        
                        # 综合分数考虑算法评分和资源权重
                        combined_rl_score = rl_score * (resource_factor + computing_factor + network_factor)
                        final_scores[node] = rl_weight * combined_rl_score + algorithm_weight * algo_score
                        
                    logger.info(f"Pod {pod_identifier}: 混合评分 (RL权重: {rl_weight:.2f}, 算法权重: {algorithm_weight:.2f})")
                    
                except Exception as e:
                    logger.error(f"强化学习评分出错: {str(e)}，回退到算法")
                    final_scores = algorithm_scores
            else:
                # 仅使用算法
                final_scores = algorithm_scores
            
            # 归一化得分
            total_score = sum(final_scores.values())
            if total_score > 0:
                final_scores = {node: score/total_score for node, score in final_scores.items()}
            
            # 转换为接口所需的格式
            host_priority_list = [
                {"Host": node, "Score": str(int(score * 100))}  # 将0-1的分数转为0-100的字符串
                for node, score in final_scores.items()
            ]
            
            # 按得分排序（从高到低）
            host_priority_list.sort(key=lambda x: x["Score"], reverse=True)
            
            # 记录最终选择的节点（得分最高的）
            if host_priority_list:
                self.last_selected_node = host_priority_list[0]['Host']
            
            # 记录日志
            best_node = host_priority_list[0]['Host'] if host_priority_list else 'None'
            logger.info(f"Pod {pod_identifier}: 节点优先级计算完成，最佳节点: {best_node}")
            
            # 如果提供了调度方案存储服务，保存调度方案
            if self.plan_store and host_priority_list:
                try:
                    # 构建调度方案
                    plan = {
                        "pod_name": pod_identifier,
                        "timestamp": datetime.now().isoformat(),
                        "selected_node": best_node,
                        "node_scores": {item["Host"]: float(item["Score"])/100 for item in host_priority_list}
                    }
                    # 保存调度方案
                    self.plan_store.store_plan(pod_identifier, plan)
                    logger.info(f"Pod {pod_identifier}: 调度方案已保存到存储服务")
                except Exception as e:
                    logger.error(f"Pod {pod_identifier}: 保存调度方案时出错: {str(e)}")
            
            # 如果RL代理初始化完成，进行训练
            if self.enable_rl and self.rl_agent and self.rl_agent.initialized:
                try:
                    loss = self.rl_agent.train()
                    if loss is not None:
                        logger.debug(f"训练强化学习代理，损失: {loss:.6f}")
                except Exception as e:
                    logger.warning(f"训练强化学习代理时出错: {str(e)}")
            
            return PriorityResponse(
                hostPriorityList=host_priority_list,
                error=""
            )
        except Exception as e:
            logger.error(f"Pod {pod_identifier}: 计算节点得分时发生错误: {str(e)}")
            # 返回空结果但不抛出异常，确保API调用不会失败
            return PriorityResponse(
                hostPriorityList=[],
                error=f"计算节点得分失败: {str(e)}"
            )
    
    def _create_rl_state(self, pod_name: str, node_features: Dict[str, List[float]], 
                        resource_info: Dict[str, Dict[str, float]]) -> RLState:
        """创建强化学习状态
        
        Args:
            pod_name: Pod名称
            node_features: 节点特征矩阵
            resource_info: 资源信息
            
        Returns:
            RLState: 强化学习状态
        """
        return RLState(
            pod_name=pod_name,
            node_features=node_features,
            resource_usage=resource_info,
            timestamp=time.time()
        )
    
    def _update_last_experience(self, current_resource_info: Dict[str, Dict[str, float]]):
        """更新上一次的经验
        
        Args:
            current_resource_info: 当前资源信息
        """
        if not self.last_state or not self.last_action or not self.last_selected_node or not self.last_node_scores:
            return
            
        try:
            # 计算奖励
            reward = self.rl_agent.calculate_reward(
                self.last_selected_node,
                self.last_node_scores,
                current_resource_info
            )
            
            # 增强奖励信号，考虑更多资源因素
            if self.last_selected_node in current_resource_info:
                node_resources = current_resource_info[self.last_selected_node]
                
                # 考虑CPU和内存使用情况
                cpu_usage = node_resources.get('cpu', 0.5)
                memory_usage = node_resources.get('memory', 0.5)
                
                # 资源均衡性奖励（CPU和内存使用率差异小则奖励高）
                balance_reward = 1.0 - abs(cpu_usage - memory_usage)
                
                # 资源效率奖励（资源使用率适中，既不过高也不过低）
                efficiency_reward = 1.0 - abs(0.5 - (cpu_usage + memory_usage) / 2)
                
                # 综合奖励，增加资源均衡性和效率因素
                reward = 0.6 * reward + 0.2 * balance_reward + 0.2 * efficiency_reward
                
                logger.debug(f"增强的奖励计算: 基础={reward:.4f}, 均衡性={balance_reward:.4f}, 效率={efficiency_reward:.4f}")
            
            # 创建经验对象
            experience = RLExperience(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=None,  # 下一个状态在下一次调度时获得
                done=False  # 连续任务，不会结束
            )
            
            # 存储经验
            self.rl_agent.store_experience(experience)
            logger.debug(f"存储经验：选择节点 {self.last_selected_node}，奖励 {reward:.4f}")
        except Exception as e:
            logger.warning(f"更新经验时出错: {str(e)}")
            
    def _genetic_algorithm(
        self,
        nodes: List[str],
        resource_info: Dict[str, Dict[str, float]],
        computing_power_info: Optional[Dict[str, Dict[str, Any]]] = None,
        network_info: Optional[Dict[str, Dict[str, Any]]] = None,
        resource_scores: Dict[str, float] = None,
        computing_scores: Dict[str, float] = None,
        network_scores: Dict[str, float] = None
    ) -> Dict[str, float]:
        """使用遗传算法计算节点得分
        
        Args:
            nodes: 节点列表
            resource_info: 资源信息
            computing_power_info: 算力信息
            network_info: 网络信息
            resource_scores: 资源得分
            computing_scores: 算力得分
            network_scores: 网络得分
            
        Returns:
            Dict[str, float]: 节点得分 {node: score}
        """
        try:
            if not nodes:
                logger.warning("节点列表为空，无法执行遗传算法")
                return {}
                
            # 确保resource_info不为空且格式正确
            if not resource_info:
                logger.warning("资源信息为空，使用默认值")
                resource_info = {node: {'cpu': 0.5, 'memory': 0.5} for node in nodes}
            
            # 计算基础数据
            if not resource_scores:
                resource_scores = self._calculate_resource_scores(nodes, resource_info)
            if not computing_scores:
                computing_scores = self._calculate_computing_scores(nodes, computing_power_info)
            if not network_scores:
                network_scores = self._calculate_network_scores(nodes, network_info)
            
            # 节点特征矩阵: 每个节点对应一个特征向量[cpu使用率, 内存使用率, 算力得分, 网络得分]
            node_features = {}
            for node in nodes:
                cpu_usage = resource_info.get(node, {}).get('cpu', 1.0)  # 默认为1.0表示资源已满
                memory_usage = resource_info.get(node, {}).get('memory', 1.0)
                
                node_features[node] = [
                    cpu_usage,  # CPU使用率，越低越好
                    memory_usage,  # 内存使用率，越低越好
                    computing_scores.get(node, 0),  # 算力得分，越高越好
                    network_scores.get(node, 0)  # 网络得分，越高越好
                ]
            
            # Step 1: 初始化种群 - 创建多个可能的节点排序方案
            population_size = self.ga_params['population_size']
            population = self._initialize_population(nodes, population_size)
            
            if not population:
                logger.warning("无法初始化种群，使用简单加权评分")
                return self._fallback_scoring(nodes, resource_scores, computing_scores, network_scores)
            
            # Step 2: 迭代优化
            best_fitness = -float('inf')
            best_individual = None
            
            for generation in range(self.ga_params['generations']):
                try:
                    # 计算适应度
                    fitness_scores = [
                        self._calculate_fitness(individual, node_features)
                        for individual in population
                    ]
                    
                    if not fitness_scores or all(math.isnan(score) for score in fitness_scores):
                        logger.warning(f"第{generation}代适应度计算失败，使用简单加权评分")
                        return self._fallback_scoring(nodes, resource_scores, computing_scores, network_scores)
                    
                    # 更新全局最优解
                    max_fitness_idx = fitness_scores.index(max(fitness_scores))
                    if fitness_scores[max_fitness_idx] > best_fitness:
                        best_fitness = fitness_scores[max_fitness_idx]
                        best_individual = population[max_fitness_idx]
                        
                    # 生成下一代
                    next_population = []
                    
                    # 精英策略: 保留一部分适应度最高的个体
                    elitism_count = int(population_size * self.ga_params['elitism_rate'])
                    if elitism_count > 0:
                        elites_idx = sorted(range(len(fitness_scores)), 
                                          key=lambda i: fitness_scores[i], 
                                          reverse=True)[:elitism_count]
                        next_population.extend([population[i] for i in elites_idx])
                    
                    # 通过选择、交叉和变异生成剩余个体
                    max_attempts = 100  # 防止无限循环
                    attempts = 0
                    while len(next_population) < population_size and attempts < max_attempts:
                        attempts += 1
                        try:
                            # 选择
                            parent1 = self._tournament_selection(population, fitness_scores)
                            parent2 = self._tournament_selection(population, fitness_scores)
                            
                            # 交叉
                            if random.random() < 0.8:  # 交叉概率
                                child1, child2 = self._crossover(parent1, parent2)
                            else:
                                child1, child2 = parent1[:], parent2[:]
                            
                            # 变异
                            child1 = self._mutate(child1, self.ga_params['mutation_rate'])
                            child2 = self._mutate(child2, self.ga_params['mutation_rate'])
                            
                            next_population.append(child1)
                            if len(next_population) < population_size:
                                next_population.append(child2)
                        except Exception as e:
                            logger.warning(f"生成下一代个体时出错: {str(e)}")
                            continue
                    
                    # 如果无法生成足够的下一代个体，填充随机个体
                    while len(next_population) < population_size:
                        individual = [random.random() for _ in range(len(nodes))]
                        total = sum(individual)
                        if total > 0:
                            individual = [w / total for w in individual]
                        next_population.append(individual)
                    
                    # 更新种群
                    population = next_population
                    
                    if generation % 10 == 0:
                        logger.debug(f"遗传算法迭代 {generation}/{self.ga_params['generations']}, 最佳适应度: {best_fitness}")
                except Exception as e:
                    logger.warning(f"遗传算法第{generation}代迭代出错: {str(e)}")
                    continue
            
            # 根据最佳个体生成节点得分
            if best_individual:
                logger.info(f"遗传算法完成，最佳适应度: {best_fitness}")
                return self._individual_to_scores(best_individual, nodes)
            else:
                # 如果遗传算法未能找到解，退化为简单加权评分
                logger.warning("遗传算法未能找到解，使用简单加权评分")
                return self._fallback_scoring(nodes, resource_scores, computing_scores, network_scores)
        except Exception as e:
            logger.error(f"遗传算法执行出错: {str(e)}")
            # 返回平均分配的得分
            return {node: 1.0/len(nodes) if nodes else 0.0 for node in nodes}
    
    def _fallback_scoring(self, nodes: List[str], resource_scores: Dict[str, float], 
                         computing_scores: Dict[str, float], network_scores: Dict[str, float]) -> Dict[str, float]:
        """当遗传算法失败时的备用评分方法
        
        Args:
            nodes: 节点列表
            resource_scores: 资源得分
            computing_scores: 算力得分
            network_scores: 网络得分
            
        Returns:
            Dict[str, float]: 节点得分 {node: score}
        """
        final_scores = {}
        for node in nodes:
            score = (
                self.weights['basic_resources'] * resource_scores.get(node, 0) +
                self.weights['computing_power'] * computing_scores.get(node, 0) +
                self.weights['network'] * network_scores.get(node, 0)
            )
            final_scores[node] = score
                
        # 归一化得分
        total_score = sum(final_scores.values())
        if total_score > 0:
            final_scores = {node: score/total_score for node, score in final_scores.items()}
        else:
            # 如果总分为0，平均分配
            final_scores = {node: 1.0/len(nodes) if nodes else 0.0 for node in nodes}
            
        return final_scores
    
    def _initialize_population(self, nodes: List[str], population_size: int) -> List[List[float]]:
        """初始化种群
        
        为每个节点生成一个权重，代表该节点被选择的概率
        
        Args:
            nodes: 节点列表
            population_size: 种群大小
            
        Returns:
            List[List[float]]: 种群，每个个体是一个节点权重列表
        """
        population = []
        for _ in range(population_size):
            # 为每个节点随机生成一个0到1之间的权重
            individual = [random.random() for _ in range(len(nodes))]
            # 归一化权重，使其总和为1
            total = sum(individual)
            if total > 0:
                individual = [w / total for w in individual]
            population.append(individual)
        return population
    
    def _calculate_fitness(self, individual: List[float], node_features: Dict[str, List[float]]) -> float:
        """计算个体适应度
        
        Args:
            individual: 个体（节点权重列表）
            node_features: 节点特征 {node: [cpu_usage, memory_usage, computing_score, network_score]}
            
        Returns:
            float: 适应度得分
        """
        try:
            nodes = list(node_features.keys())
            if not nodes or not individual or len(individual) != len(nodes):
                logger.warning(f"无效的个体或节点特征: 个体长度={len(individual) if individual else 0}, 节点数={len(nodes)}")
                return -float('inf')  # 返回极小值表示这是一个无效解
            
            # 资源均衡性
            resource_balance = 0
            # 资源利用率
            resource_utilization = 0
            # 算力得分
            computing_score = 0
            # 网络得分
            network_score = 0
            
            for i, node in enumerate(nodes):
                if i >= len(individual):
                    break
                    
                weight = individual[i]
                if weight < 0 or math.isnan(weight) or math.isinf(weight):
                    weight = 0.0  # 处理无效权重
                    
                features = node_features.get(node, [0.5, 0.5, 0, 0])
                if len(features) < 4:
                    # 确保特征向量完整
                    features = features + [0] * (4 - len(features))
                
                # CPU和内存使用率（越低越好）
                cpu_usage = features[0]
                if cpu_usage < 0 or cpu_usage > 1 or math.isnan(cpu_usage) or math.isinf(cpu_usage):
                    cpu_usage = 0.5  # 使用默认值
                    
                memory_usage = features[1]
                if memory_usage < 0 or memory_usage > 1 or math.isnan(memory_usage) or math.isinf(memory_usage):
                    memory_usage = 0.5  # 使用默认值
                
                # 资源均衡性（CPU和内存使用率的标准差，越小越均衡）
                try:
                    resource_std = np.std([cpu_usage, memory_usage])
                    resource_balance -= resource_std * weight  # 负值，因为我们要最大化适应度
                except Exception as e:
                    logger.warning(f"计算资源均衡性时出错: {str(e)}")
                    resource_balance -= 0.5 * weight  # 使用默认值
                
                # 资源利用率（CPU和内存的平均使用率，越低越好）
                resource_avg = (cpu_usage + memory_usage) / 2
                resource_utilization += (1 - resource_avg) * weight
                
                # 算力得分（直接使用计算好的算力得分，越高越好）
                computing_feature = features[2]
                if math.isnan(computing_feature) or math.isinf(computing_feature):
                    computing_feature = 0  # 使用默认值
                computing_score += computing_feature * weight
                
                # 网络得分（直接使用计算好的网络得分，越高越好）
                network_feature = features[3]
                if math.isnan(network_feature) or math.isinf(network_feature):
                    network_feature = 0  # 使用默认值
                network_score += network_feature * weight
            
            # 获取权重参数，确保它们有效
            weight_resource_utilization = self.ga_params.get('weight_resource_utilization', 0.25)
            weight_resource_balance = self.ga_params.get('weight_resource_balance', 0.25)
            weight_computing = self.ga_params.get('weight_computing', 0.25)
            weight_network = self.weights.get('network', 0.25)
            
            # 总适应度 = 各项指标的加权和，根据算法类型选择适当的权重
            if self.optimization_method == 'pso':
                weight_resource_utilization = self.pso_params.get('weight_resource_utilization', 0.25)
                weight_resource_balance = self.pso_params.get('weight_resource_balance', 0.25)
                weight_computing = self.pso_params.get('weight_computing', 0.25)
                weight_network = self.pso_params.get('weight_network', 0.25)
            elif self.optimization_method == 'sa':
                weight_resource_utilization = self.sa_params.get('weight_resource_utilization', 0.25)
                weight_resource_balance = self.sa_params.get('weight_resource_balance', 0.25)
                weight_computing = self.sa_params.get('weight_computing', 0.25)
                weight_network = self.sa_params.get('weight_network', 0.25)
            elif self.optimization_method == 'aco':
                weight_resource_utilization = self.aco_params.get('weight_resource_utilization', 0.25)
                weight_resource_balance = self.aco_params.get('weight_resource_balance', 0.25)
                weight_computing = self.aco_params.get('weight_computing', 0.25)
                weight_network = self.aco_params.get('weight_network', 0.25)
            
            fitness = (
                weight_resource_utilization * resource_utilization +
                weight_resource_balance * resource_balance +
                weight_computing * computing_score +
                weight_network * network_score
            )
            
            # 检查适应度是否有效
            if math.isnan(fitness) or math.isinf(fitness):
                logger.warning(f"计算出无效的适应度: {fitness}")
                return -float('inf')  # 返回极小值表示这是一个无效解
                
            return fitness
        except Exception as e:
            logger.warning(f"计算适应度时出错: {str(e)}")
            return -float('inf')  # 返回极小值表示这是一个无效解
    
    def _tournament_selection(self, population: List[List[float]], fitness_scores: List[float]) -> List[float]:
        """锦标赛选择
        
        Args:
            population: 种群
            fitness_scores: 适应度得分列表
            
        Returns:
            List[float]: 选择的个体
        """
        try:
            if not population or not fitness_scores or len(population) != len(fitness_scores):
                logger.warning(f"无效的种群或适应度列表: 种群大小={len(population) if population else 0}, 适应度列表大小={len(fitness_scores) if fitness_scores else 0}")
                # 如果无法进行选择，返回一个随机个体
                if population:
                    return random.choice(population)
                else:
                    return []
                
            # 随机选择k个个体，选择其中适应度最高的
            k = min(3, len(population))  # 锦标赛大小，确保不超过种群大小
            if k == 0:
                return []
                
            selected_indices = random.sample(range(len(population)), k)
            
            # 找出适应度最高的个体
            valid_indices = [i for i in selected_indices if i < len(fitness_scores) and not math.isnan(fitness_scores[i]) and not math.isinf(fitness_scores[i])]
            if not valid_indices:
                # 如果没有有效的个体，随机选择一个
                if population:
                    return random.choice(population)
                else:
                    return []
                    
            selected_index = max(valid_indices, key=lambda i: fitness_scores[i])
            return population[selected_index]
        except Exception as e:
            logger.warning(f"锦标赛选择出错: {str(e)}")
            # 如果出错，返回一个随机个体
            if population:
                return random.choice(population)
            else:
                return []
    
    def _crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """交叉操作
        
        Args:
            parent1: 父个体1
            parent2: 父个体2
            
        Returns:
            Tuple[List[float], List[float]]: 两个子个体
        """
        try:
            # 检查父个体是否有效
            if not parent1 or not parent2:
                logger.warning("无效的父个体")
                # 返回父个体的副本
                return parent1[:] if parent1 else [], parent2[:] if parent2 else []
                
            # 确保父个体长度相同
            min_len = min(len(parent1), len(parent2))
            if min_len == 0:
                return [], []
                
            parent1 = parent1[:min_len]
            parent2 = parent2[:min_len]
            
            # 单点交叉
            crossover_point = random.randint(1, min_len - 1) if min_len > 1 else 0
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            
            # 归一化
            total1 = sum(child1)
            total2 = sum(child2)
            if total1 > 0:
                child1 = [w / total1 for w in child1]
            else:
                # 如果总和为0，使用均匀分布
                child1 = [1.0 / len(child1) for _ in range(len(child1))] if child1 else []
                
            if total2 > 0:
                child2 = [w / total2 for w in child2]
            else:
                # 如果总和为0，使用均匀分布
                child2 = [1.0 / len(child2) for _ in range(len(child2))] if child2 else []
                
            return child1, child2
        except Exception as e:
            logger.warning(f"交叉操作出错: {str(e)}")
            # 如果出错，返回父个体的副本
            return parent1[:] if parent1 else [], parent2[:] if parent2 else []
    
    def _mutate(self, individual: List[float], mutation_rate: float) -> List[float]:
        """变异操作
        
        Args:
            individual: 个体
            mutation_rate: 变异率
            
        Returns:
            List[float]: 变异后的个体
        """
        try:
            if not individual:
                logger.warning("无效的个体")
                return []
                
            # 确保变异率在有效范围内
            mutation_rate = max(0.0, min(1.0, mutation_rate))
            
            # 创建个体的副本进行变异
            mutated = individual[:]
            
            for i in range(len(mutated)):
                if random.random() < mutation_rate:
                    # 生成一个随机偏移量
                    delta = random.uniform(-0.1, 0.1)
                    mutated[i] = max(0.0, min(1.0, mutated[i] + delta))
            
            # 归一化
            total = sum(mutated)
            if total > 0:
                mutated = [w / total for w in mutated]
            else:
                # 如果总和为0，使用均匀分布
                mutated = [1.0 / len(mutated) for _ in range(len(mutated))] if mutated else []
                
            return mutated
        except Exception as e:
            logger.warning(f"变异操作出错: {str(e)}")
            # 如果出错，返回原始个体
            return individual
    
    def _individual_to_scores(self, individual: List[float], nodes: List[str]) -> Dict[str, float]:
        """将个体转换为节点得分
        
        Args:
            individual: 个体（节点权重列表）
            nodes: 节点列表
            
        Returns:
            Dict[str, float]: 节点得分 {node: score}
        """
        try:
            if not individual or not nodes:
                logger.warning(f"无效的个体或节点列表: 个体长度={len(individual) if individual else 0}, 节点数={len(nodes) if nodes else 0}")
                # 如果无法转换，返回均匀分布的得分
                return {node: 1.0/len(nodes) if nodes else 0.0 for node in nodes}
                
            scores = {}
            # 确保个体和节点列表长度匹配
            min_len = min(len(individual), len(nodes))
            
            for i in range(min_len):
                node = nodes[i]
                score = individual[i]
                
                # 确保得分有效
                if math.isnan(score) or math.isinf(score) or score < 0:
                    score = 0.0
                    
                scores[node] = score
                
            # 处理剩余节点（如果有）
            for i in range(min_len, len(nodes)):
                scores[nodes[i]] = 0.0
                
            # 归一化得分
            total_score = sum(scores.values())
            if total_score > 0:
                scores = {node: score/total_score for node, score in scores.items()}
            else:
                # 如果总分为0，平均分配
                scores = {node: 1.0/len(nodes) if nodes else 0.0 for node in nodes}
                
            return scores
        except Exception as e:
            logger.warning(f"将个体转换为节点得分时出错: {str(e)}")
            # 如果出错，返回均匀分布的得分
            return {node: 1.0/len(nodes) if nodes else 0.0 for node in nodes}
            
    def _calculate_resource_scores(self, nodes: List[str], resource_info: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """计算基础资源得分
        
        Args:
            nodes: 节点列表
            resource_info: 资源信息 {node: {'cpu': usage, 'memory': usage}}
            
        Returns:
            Dict[str, float]: 资源得分 {node: score}
        """
        scores = {}
        for node in nodes:
            cpu_usage = resource_info.get(node, {}).get('cpu', 1.0)
            memory_usage = resource_info.get(node, {}).get('memory', 1.0)
            
            # 计算得分（资源使用率越低得分越高）
            cpu_score = 1.0 - cpu_usage
            memory_score = 1.0 - memory_usage
            
            # 综合得分（0-1范围）
            score = (cpu_score + memory_score) / 2
            scores[node] = score
            
        return scores
        
    def _calculate_computing_scores(self, nodes: List[str], computing_power_info: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, float]:
        """计算算力得分
        
        Args:
            nodes: 节点列表
            computing_power_info: 算力信息 {node: {'cpu_load': load, 'gpu_load': load}}
            
        Returns:
            Dict[str, float]: 算力得分 {node: score}
        """
        if not computing_power_info:
            # 如果没有算力信息，返回默认得分
            return {node: 0.5 for node in nodes}
            
        scores = {}
        for node in nodes:
            # 获取节点算力负载，默认为高负载
            cpu_load = computing_power_info.get(node, {}).get('cpu_load', 0.8)
            gpu_load = computing_power_info.get(node, {}).get('gpu_load', 0.8)
            
            # 算力值（算力负载越低得分越高）
            cpu_power_score = 1.0 - cpu_load
            gpu_power_score = 1.0 - gpu_load
            
            # 综合算力得分（0-1范围）
            if 'gpu_available' in computing_power_info.get(node, {}) and computing_power_info[node]['gpu_available']:
                # 如果有GPU，考虑GPU得分
                score = 0.6 * cpu_power_score + 0.4 * gpu_power_score
            else:
                # 如果没有GPU，只考虑CPU得分
                score = cpu_power_score
            
            scores[node] = score
            
        return scores
        
    def _calculate_network_scores(self, nodes: List[str], network_info: Optional[Dict[str, Dict[str, Any]]]) -> Dict[str, float]:
        """计算网络得分
        
        Args:
            nodes: 节点列表
            network_info: 网络信息 {node: {'latency': ms, 'bandwidth': mbps}}
            
        Returns:
            Dict[str, float]: 网络得分 {node: score}
        """
        if not network_info:
            # 如果没有网络信息，返回默认得分
            return {node: 0.5 for node in nodes}
            
        # 提取所有节点的延迟和带宽值
        latencies = []
        bandwidths = []
        
        for node in nodes:
            node_network = network_info.get(node, {})
            if 'latency' in node_network:
                latencies.append(node_network['latency'])
            if 'bandwidth' in node_network:
                bandwidths.append(node_network['bandwidth'])
                
        # 计算延迟和带宽的最大最小值（用于归一化）
        max_latency = max(latencies) if latencies else 1.0
        min_latency = min(latencies) if latencies else 0.0
        max_bandwidth = max(bandwidths) if bandwidths else 1.0
        min_bandwidth = min(bandwidths) if bandwidths else 0.0
        
        # 计算网络得分
        scores = {}
        for node in nodes:
            node_network = network_info.get(node, {})
            
            # 归一化延迟（延迟越低越好）
            latency = node_network.get('latency', max_latency)
            norm_latency = 1.0 - (latency - min_latency) / (max_latency - min_latency) if max_latency > min_latency else 0.5
            
            # 归一化带宽（带宽越高越好）
            bandwidth = node_network.get('bandwidth', min_bandwidth)
            norm_bandwidth = (bandwidth - min_bandwidth) / (max_bandwidth - min_bandwidth) if max_bandwidth > min_bandwidth else 0.5
            
            # 综合网络得分（0-1范围）
            score = 0.6 * norm_latency + 0.4 * norm_bandwidth
            scores[node] = score
            
        return scores

    def _particle_swarm_optimization(
        self,
        nodes: List[str],
        resource_info: Dict[str, Dict[str, float]],
        computing_power_info: Optional[Dict[str, Dict[str, Any]]] = None,
        network_info: Optional[Dict[str, Dict[str, Any]]] = None,
        resource_scores: Dict[str, float] = None,
        computing_scores: Dict[str, float] = None,
        network_scores: Dict[str, float] = None
    ) -> Dict[str, float]:
        """使用粒子群优化算法计算节点得分
        
        Args:
            nodes: 节点列表
            resource_info: 资源信息
            computing_power_info: 算力信息
            network_info: 网络信息
            resource_scores: 资源得分
            computing_scores: 算力得分
            network_scores: 网络得分
            
        Returns:
            Dict[str, float]: 节点得分 {node: score}
        """
        try:
            if not nodes:
                logger.warning("节点列表为空，无法执行粒子群优化算法")
                return {}
                
            # 确保resource_info不为空且格式正确
            if not resource_info:
                logger.warning("资源信息为空，使用默认值")
                resource_info = {node: {'cpu': 0.5, 'memory': 0.5} for node in nodes}
            
            # 计算基础数据
            if not resource_scores:
                resource_scores = self._calculate_resource_scores(nodes, resource_info)
            if not computing_scores:
                computing_scores = self._calculate_computing_scores(nodes, computing_power_info)
            if not network_scores:
                network_scores = self._calculate_network_scores(nodes, network_info)
            
            # 节点特征矩阵: 每个节点对应一个特征向量[cpu使用率, 内存使用率, 算力得分, 网络得分]
            node_features = {}
            for node in nodes:
                cpu_usage = resource_info.get(node, {}).get('cpu', 1.0)  # 默认为1.0表示资源已满
                memory_usage = resource_info.get(node, {}).get('memory', 1.0)
                
                node_features[node] = [
                    cpu_usage,  # CPU使用率，越低越好
                    memory_usage,  # 内存使用率，越低越好
                    computing_scores.get(node, 0),  # 算力得分，越高越好
                    network_scores.get(node, 0)  # 网络得分，越高越好
                ]
            
            # Step 1: 初始化粒子群
            swarm_size = self.pso_params['swarm_size']
            swarm = self._initialize_swarm(nodes, swarm_size)
            
            if not swarm:
                logger.warning("无法初始化粒子群，使用简单加权评分")
                return self._fallback_scoring(nodes, resource_scores, computing_scores, network_scores)
            
            # 初始化每个粒子的最佳位置和全局最佳位置
            particle_best_positions = swarm.copy()  # 个体最优位置，初始为当前位置
            particle_best_fitness = [self._calculate_fitness(particle, node_features) for particle in swarm]
            
            # 全局最优位置
            global_best_index = particle_best_fitness.index(max(particle_best_fitness)) if particle_best_fitness else 0
            global_best_position = swarm[global_best_index].copy() if swarm else []
            global_best_fitness = particle_best_fitness[global_best_index] if particle_best_fitness else -float('inf')
            
            # 初始化粒子速度
            velocities = []
            for _ in range(swarm_size):
                velocity = [random.uniform(-0.1, 0.1) for _ in range(len(nodes))]
                velocities.append(velocity)
            
            # Step 2: 迭代优化
            max_iterations = self.pso_params['max_iterations']
            w = self.pso_params['inertia_weight']  # 惯性权重
            c1 = self.pso_params['cognitive_coef']  # 认知系数
            c2 = self.pso_params['social_coef']  # 社会系数
            
            logger.debug(f"PSO参数: 粒子数={swarm_size}, 迭代次数={max_iterations}, "
                        f"惯性权重={w}, 认知系数={c1}, 社会系数={c2}")
            
            for iteration in range(max_iterations):
                try:
                    # 更新每个粒子的位置和速度
                    for i in range(len(swarm)):
                        particle = swarm[i]
                        velocity = velocities[i]
                        
                        # 计算新速度
                        for j in range(len(velocity)):
                            r1 = random.random()
                            r2 = random.random()
                            
                            # 速度更新公式
                            # v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
                            cognitive_velocity = c1 * r1 * (particle_best_positions[i][j] - particle[j])
                            social_velocity = c2 * r2 * (global_best_position[j] - particle[j])
                            
                            velocity[j] = w * velocity[j] + cognitive_velocity + social_velocity
                            # 限制速度范围
                            velocity[j] = max(-0.5, min(0.5, velocity[j]))
                        
                        # 更新位置
                        for j in range(len(particle)):
                            particle[j] += velocity[j]
                            # 确保位置在有效范围内
                            particle[j] = max(0.0, min(1.0, particle[j]))
                        
                        # 归一化位置（确保总和为1）
                        total = sum(particle)
                        if total > 0:
                            particle = [p / total for p in particle]
                            swarm[i] = particle
                        
                        # 计算适应度
                        fitness = self._calculate_fitness(particle, node_features)
                        
                        # 更新个体最佳位置
                        if fitness > particle_best_fitness[i]:
                            particle_best_positions[i] = particle.copy()
                            particle_best_fitness[i] = fitness
                            
                            # 更新全局最佳位置
                            if fitness > global_best_fitness:
                                global_best_position = particle.copy()
                                global_best_fitness = fitness
                
                    if iteration % 10 == 0:
                        logger.debug(f"PSO迭代 {iteration}/{max_iterations}, 最佳适应度: {global_best_fitness}")
                except Exception as e:
                    logger.warning(f"PSO第{iteration}次迭代出错: {str(e)}")
                    continue
            
            # 根据全局最佳位置生成节点得分
            if global_best_position:
                logger.info(f"PSO算法完成，最佳适应度: {global_best_fitness}")
                return self._individual_to_scores(global_best_position, nodes)
            else:
                # 如果PSO未能找到解，退化为简单加权评分
                logger.warning("PSO算法未能找到解，使用简单加权评分")
                return self._fallback_scoring(nodes, resource_scores, computing_scores, network_scores)
        except Exception as e:
            logger.error(f"PSO算法执行出错: {str(e)}")
            # 返回平均分配的得分
            return {node: 1.0/len(nodes) if nodes else 0.0 for node in nodes}
    
    def _initialize_swarm(self, nodes: List[str], swarm_size: int) -> List[List[float]]:
        """初始化粒子群
        
        为每个节点生成一个权重，代表该节点被选择的概率
        
        Args:
            nodes: 节点列表
            swarm_size: 粒子群大小
            
        Returns:
            List[List[float]]: 粒子群，每个粒子是一个节点权重列表
        """
        swarm = []
        for _ in range(swarm_size):
            # 为每个节点随机生成一个0到1之间的权重
            particle = [random.random() for _ in range(len(nodes))]
            # 归一化权重，使其总和为1
            total = sum(particle)
            if total > 0:
                particle = [w / total for w in particle]
            swarm.append(particle)
        return swarm

    def _simulated_annealing(
        self,
        nodes: List[str],
        resource_info: Dict[str, Dict[str, float]],
        computing_power_info: Optional[Dict[str, Dict[str, Any]]] = None,
        network_info: Optional[Dict[str, Dict[str, Any]]] = None,
        resource_scores: Dict[str, float] = None,
        computing_scores: Dict[str, float] = None,
        network_scores: Dict[str, float] = None
    ) -> Dict[str, float]:
        """使用模拟退火算法计算节点得分
        
        Args:
            nodes: 节点列表
            resource_info: 资源信息
            computing_power_info: 算力信息
            network_info: 网络信息
            resource_scores: 资源得分
            computing_scores: 算力得分
            network_scores: 网络得分
            
        Returns:
            Dict[str, float]: 节点得分 {node: score}
        """
        try:
            if not nodes:
                logger.warning("节点列表为空，无法执行模拟退火算法")
                return {}
                
            # 确保resource_info不为空且格式正确
            if not resource_info:
                logger.warning("资源信息为空，使用默认值")
                resource_info = {node: {'cpu': 0.5, 'memory': 0.5} for node in nodes}
            
            # 计算基础数据
            if not resource_scores:
                resource_scores = self._calculate_resource_scores(nodes, resource_info)
            if not computing_scores:
                computing_scores = self._calculate_computing_scores(nodes, computing_power_info)
            if not network_scores:
                network_scores = self._calculate_network_scores(nodes, network_info)
            
            # 节点特征矩阵: 每个节点对应一个特征向量[cpu使用率, 内存使用率, 算力得分, 网络得分]
            node_features = {}
            for node in nodes:
                cpu_usage = resource_info.get(node, {}).get('cpu', 1.0)  # 默认为1.0表示资源已满
                memory_usage = resource_info.get(node, {}).get('memory', 1.0)
                
                node_features[node] = [
                    cpu_usage,  # CPU使用率，越低越好
                    memory_usage,  # 内存使用率，越低越好
                    computing_scores.get(node, 0),  # 算力得分，越高越好
                    network_scores.get(node, 0)  # 网络得分，越高越好
                ]
            
            # 初始化温度
            initial_temp = self.sa_params['initial_temp']
            cooling_rate = self.sa_params['cooling_rate']
            min_temp = self.sa_params['min_temp']
            iterations_per_temp = self.sa_params['iterations_per_temp']
            
            # 初始化当前解
            current_solution = [random.random() for _ in range(len(nodes))]
            current_score = self._calculate_fitness(current_solution, node_features)
            
            # 初始化最佳解
            best_solution = current_solution.copy()
            best_score = current_score
            
            # 模拟退火算法
            temperature = initial_temp
            while temperature > min_temp:
                for _ in range(iterations_per_temp):
                    # 生成新解
                    new_solution = self._mutate(current_solution, 0.1)
                    new_score = self._calculate_fitness(new_solution, node_features)
                    
                    # 计算能量差
                    delta_score = new_score - current_score
                    
                    # 接受新解的概率
                    acceptance_prob = math.exp(delta_score / temperature)
                    
                    if delta_score > 0 or random.random() < acceptance_prob:
                        current_solution = new_solution.copy()
                        current_score = new_score
                        
                        if current_score > best_score:
                            best_solution = current_solution.copy()
                            best_score = current_score
                
                # 降低温度
                temperature *= cooling_rate
            
            # 根据最佳解生成节点得分
            if best_solution:
                logger.info(f"模拟退火算法完成，最佳适应度: {best_score}")
                return self._individual_to_scores(best_solution, nodes)
            else:
                # 如果模拟退火算法未能找到解，退化为简单加权评分
                logger.warning("模拟退火算法未能找到解，使用简单加权评分")
                return self._fallback_scoring(nodes, resource_scores, computing_scores, network_scores)
        except Exception as e:
            logger.error(f"模拟退火算法执行出错: {str(e)}")
            # 返回平均分配的得分
            return {node: 1.0/len(nodes) if nodes else 0.0 for node in nodes}

    def _ant_colony_optimization(
        self,
        nodes: List[str],
        resource_info: Dict[str, Dict[str, float]],
        computing_power_info: Optional[Dict[str, Dict[str, Any]]] = None,
        network_info: Optional[Dict[str, Dict[str, Any]]] = None,
        resource_scores: Dict[str, float] = None,
        computing_scores: Dict[str, float] = None,
        network_scores: Dict[str, float] = None
    ) -> Dict[str, float]:
        """使用蚁群算法计算节点得分
        
        Args:
            nodes: 节点列表
            resource_info: 资源信息
            computing_power_info: 算力信息
            network_info: 网络信息
            resource_scores: 资源得分
            computing_scores: 算力得分
            network_scores: 网络得分
            
        Returns:
            Dict[str, float]: 节点得分 {node: score}
        """
        try:
            if not nodes:
                logger.warning("节点列表为空，无法执行蚁群算法")
                return {}
                
            # 确保resource_info不为空且格式正确
            if not resource_info:
                logger.warning("资源信息为空，使用默认值")
                resource_info = {node: {'cpu': 0.5, 'memory': 0.5} for node in nodes}
            
            # 计算基础数据
            if not resource_scores:
                resource_scores = self._calculate_resource_scores(nodes, resource_info)
            if not computing_scores:
                computing_scores = self._calculate_computing_scores(nodes, computing_power_info)
            if not network_scores:
                network_scores = self._calculate_network_scores(nodes, network_info)
            
            # 节点特征矩阵: 每个节点对应一个特征向量[cpu使用率, 内存使用率, 算力得分, 网络得分]
            node_features = {}
            for node in nodes:
                cpu_usage = resource_info.get(node, {}).get('cpu', 1.0)  # 默认为1.0表示资源已满
                memory_usage = resource_info.get(node, {}).get('memory', 1.0)
                
                node_features[node] = [
                    cpu_usage,  # CPU使用率，越低越好
                    memory_usage,  # 内存使用率，越低越好
                    computing_scores.get(node, 0),  # 算力得分，越高越好
                    network_scores.get(node, 0)  # 网络得分，越高越好
                ]
            
            # 获取算法参数
            ants_count = self.aco_params['ants_count']
            iterations = self.aco_params['iterations']
            alpha = self.aco_params['alpha']  # 信息素重要程度系数
            beta = self.aco_params['beta']    # 启发函数重要程度系数
            rho = self.aco_params['rho']      # 信息素挥发系数
            q = self.aco_params['q']          # 信息素增加强度系数
            
            # 计算启发式信息(优先选择资源占用少的节点)
            heuristic_info = {}
            for node in nodes:
                features = node_features.get(node, [0.5, 0.5, 0, 0])
                # 资源使用率越低越好
                cpu_usage = features[0]
                memory_usage = features[1]
                # 算力和网络得分越高越好
                computing_score = features[2]
                network_score = features[3]
                
                # 综合启发函数值(越高越好)
                heuristic_value = (1 - (cpu_usage + memory_usage) / 2) + (computing_score + network_score) / 2
                heuristic_info[node] = max(0.001, heuristic_value)  # 确保启发值为正
            
            # 初始化信息素矩阵(所有节点间信息素初始为同一值)
            pheromone = {}
            for node1 in nodes:
                pheromone[node1] = {}
                for node2 in nodes:
                    pheromone[node1][node2] = 1.0  # 初始信息素值
            
            # 初始化最佳解
            best_solution = None
            best_fitness = -float('inf')
            
            # 迭代优化
            for iteration in range(iterations):
                try:
                    # 初始化蚂蚁
                    ants = []
                    for _ in range(ants_count):
                        ant = Ant(nodes)
                        ants.append(ant)
                    
                    # 每只蚂蚁构建解
                    for ant in ants:
                        # 随机选择起始节点
                        current_node = random.choice(nodes)
                        visited = {current_node}
                        ant.path = [current_node]
                        
                        # 按顺序选择剩余节点
                        while len(visited) < len(nodes):
                            next_node = self._select_next_node_aco(
                                current_node, 
                                [n for n in nodes if n not in visited], 
                                pheromone, 
                                heuristic_info,
                                alpha,
                                beta
                            )
                            ant.path.append(next_node)
                            visited.add(next_node)
                            current_node = next_node
                        
                        # 转换路径为节点权重
                        weights = [1.0 / len(nodes)] * len(nodes)
                        # 可以根据路径顺序设置权重，前面的节点权重更高
                        for i, node in enumerate(ant.path):
                            node_idx = nodes.index(node)
                            weights[node_idx] = 1.0 - (i / len(nodes))
                        
                        # 归一化权重
                        total = sum(weights)
                        if total > 0:
                            weights = [w / total for w in weights]
                        
                        ant.weights = weights
                        ant.fitness = self._calculate_fitness(weights, node_features)
                        
                        # 更新最佳解
                        if ant.fitness > best_fitness:
                            best_fitness = ant.fitness
                            best_solution = ant.weights.copy()
                    
                    # 更新信息素
                    self._update_pheromone(pheromone, ants, nodes, rho, q)
                    
                    if iteration % 10 == 0:
                        logger.debug(f"ACO迭代 {iteration}/{iterations}, 最佳适应度: {best_fitness}")
                except Exception as e:
                    logger.warning(f"ACO第{iteration}次迭代出错: {str(e)}")
                    continue
            
            # 根据最佳解生成节点得分
            if best_solution:
                logger.info(f"蚁群算法完成，最佳适应度: {best_fitness}")
                return self._individual_to_scores(best_solution, nodes)
            else:
                # 如果蚁群算法未能找到解，退化为简单加权评分
                logger.warning("蚁群算法未能找到解，使用简单加权评分")
                return self._fallback_scoring(nodes, resource_scores, computing_scores, network_scores)
        except Exception as e:
            logger.error(f"蚁群算法执行出错: {str(e)}")
            # 返回平均分配的得分
            return {node: 1.0/len(nodes) if nodes else 0.0 for node in nodes}
    
    def _select_next_node_aco(self, current_node: str, candidate_nodes: List[str], 
                            pheromone: Dict[str, Dict[str, float]], 
                            heuristic_info: Dict[str, float],
                            alpha: float, beta: float) -> str:
        """选择下一个节点(ACO)
        
        Args:
            current_node: 当前节点
            candidate_nodes: 候选节点列表
            pheromone: 信息素矩阵
            heuristic_info: 启发式信息
            alpha: 信息素重要程度系数
            beta: 启发函数重要程度系数
            
        Returns:
            str: 下一个节点
        """
        if not candidate_nodes:
            return current_node
        
        # 计算概率
        total = 0.0
        probabilities = {}
        
        for node in candidate_nodes:
            # 信息素值
            tau = pheromone[current_node].get(node, 0.0001)
            # 启发函数值
            eta = heuristic_info.get(node, 0.0001)
            
            # 计算转移概率的分子部分: τ^α * η^β
            probabilities[node] = pow(tau, alpha) * pow(eta, beta)
            total += probabilities[node]
        
        # 归一化概率
        if total > 0:
            for node in probabilities:
                probabilities[node] /= total
        else:
            # 如果总概率为0，使用均匀分布
            for node in candidate_nodes:
                probabilities[node] = 1.0 / len(candidate_nodes)
        
        # 轮盘赌选择
        r = random.random()
        cumulative_prob = 0.0
        
        for node, prob in probabilities.items():
            cumulative_prob += prob
            if r <= cumulative_prob:
                return node
        
        # 如果没有选中任何节点(浮点数精度问题可能导致)，随机选择一个
        return random.choice(candidate_nodes)
    
    def _update_pheromone(self, pheromone: Dict[str, Dict[str, float]], 
                        ants: List[Ant], nodes: List[str], 
                        rho: float, q: float) -> None:
        """更新信息素矩阵
        
        Args:
            pheromone: 信息素矩阵
            ants: 蚂蚁列表
            nodes: 节点列表
            rho: 信息素挥发系数
            q: 信息素增加强度系数
        """
        # 信息素挥发
        for i in pheromone:
            for j in pheromone[i]:
                pheromone[i][j] *= (1 - rho)
        
        # 信息素沉积
        for ant in ants:
            # 信息素增量与适应度成正比
            delta = q * ant.fitness if ant.fitness > 0 else 0
            
            # 在蚂蚁走过的路径上增加信息素
            for i in range(len(ant.path) - 1):
                node1 = ant.path[i]
                node2 = ant.path[i + 1]
                pheromone[node1][node2] += delta
                pheromone[node2][node1] += delta  # 考虑无向图
