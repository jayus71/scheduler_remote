"""联合任务规划服务模块"""
import json
import yaml
import os
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict, deque
from kubernetes.client import V1Pod, V1ObjectMeta, V1PodSpec, V1Container, V1ResourceRequirements
from kubernetes.client.rest import ApiException
from loguru import logger

from app.schemas.task_planning import Task, JointTasksPlan, TaskPlanningResponse
from app.core.k8s_config import get_k8s_clients


class TaskPlanningService:
    """联合任务规划服务类"""

    def __init__(self):
        """初始化Kubernetes客户端"""
        self.core_v1, _ = get_k8s_clients()

    async def parse_tasks_file(self, file_content: bytes, file_extension: str) -> JointTasksPlan:
        """
        解析任务文件内容
        
        Args:
            file_content: 文件内容
            file_extension: 文件扩展名
            
        Returns:
            JointTasksPlan: 解析后的联合任务规划
            
        Raises:
            ValueError: 当文件格式不支持或解析失败时
        """
        try:
            if file_extension.lower() == '.json':
                data = json.loads(file_content)
            elif file_extension.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(file_content)
            else:
                raise ValueError(f"不支持的文件格式: {file_extension}")
            
            # 预处理：收集有效的任务ID和记录修正的依赖关系
            modified_dependencies = {}
            
            # 先收集所有有效的任务ID
            if 'tasks' in data:
                task_ids = {task.get('task_id') for task in data['tasks'] if 'task_id' in task}
                
                # 然后检查和修复依赖关系
                for task in data['tasks']:
                    task_id = task.get('task_id')
                    if 'dependencies' in task and task['dependencies']:
                        original_deps = list(task['dependencies'])
                        valid_deps = [dep for dep in original_deps if dep in task_ids]
                        
                        # 如果有无效依赖，记录并修正
                        if len(valid_deps) != len(original_deps):
                            invalid_deps = set(original_deps) - set(valid_deps)
                            logger.warning(f"任务 {task_id} 包含无效依赖: {invalid_deps}，这些依赖将被移除")
                            task['dependencies'] = valid_deps
                            modified_dependencies[task_id] = list(invalid_deps)
            
            # 验证并转换为JointTasksPlan模型
            plan = JointTasksPlan(**data)
            
            # 将修正的依赖关系信息添加到响应中
            if modified_dependencies:
                # 使用自定义属性存储修改的依赖
                plan.__dict__['modified_dependencies'] = modified_dependencies
                logger.info(f"已修正 {len(modified_dependencies)} 个任务的依赖关系")
            
            return plan
            
        except json.JSONDecodeError:
            logger.error("JSON解析错误")
            raise ValueError("无效的JSON格式")
        except yaml.YAMLError:
            logger.error("YAML解析错误")
            raise ValueError("无效的YAML格式")
        except Exception as e:
            logger.error(f"解析任务文件失败: {str(e)}")
            raise ValueError(f"解析任务文件失败: {str(e)}")

    async def topological_sort(self, plan: JointTasksPlan) -> Tuple[List[str], Dict[str, Task]]:
        """
        对任务进行拓扑排序
        
        Args:
            plan: 联合任务规划
            
        Returns:
            Tuple[List[str], Dict[str, Task]]: 排序后的任务ID列表和任务字典
            
        Raises:
            ValueError: 当任务依赖关系存在循环时
        """
        # 构建任务字典和邻接表
        task_dict = {task.task_id: task for task in plan.tasks}
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        # 收集所有任务节点（包括那些没有明确出现在依赖列表中的）
        all_nodes = set(task_dict.keys())
        
        # 构建图和计算入度
        for task in plan.tasks:
            task_id = task.task_id
            
            # 确保每个任务都在入度字典中
            if task_id not in in_degree:
                in_degree[task_id] = 0
            
            # 验证并添加依赖边
            valid_deps = []
            for dep in task.dependencies:
                if dep not in task_dict:
                    logger.warning(f"任务 {task_id} 依赖不存在的任务 {dep}，将忽略此依赖")
                    continue
                valid_deps.append(dep)
                graph[dep].append(task_id)
                in_degree[task_id] += 1
            
            # 更新任务的有效依赖
            if len(valid_deps) != len(task.dependencies):
                task.dependencies = valid_deps
        
        # Kahn算法进行拓扑排序
        result = []
        
        # 按字母顺序选择入度为0的节点，确保排序的确定性
        zero_in_degree = sorted([node for node, degree in in_degree.items() if degree == 0])
        queue = deque(zero_in_degree)
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            # 获取所有邻居并按ID排序，确保确定性
            neighbors = sorted(graph[current])
            for neighbor in neighbors:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 检查是否有循环依赖
        if len(result) != len(all_nodes):
            unprocessed = all_nodes - set(result)
            cycles = self._find_cycles(graph, unprocessed)
            cycle_info = ", ".join([" -> ".join(cycle) for cycle in cycles])
            error_msg = f"任务依赖关系存在循环，无法进行拓扑排序。检测到以下环路: {cycle_info}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"成功完成拓扑排序: {' -> '.join(result)}")
        return result, task_dict
        
    def _find_cycles(self, graph: Dict[str, List[str]], start_nodes: Set[str]) -> List[List[str]]:
        """
        查找有向图中的环路
        
        Args:
            graph: 有向图（邻接表）
            start_nodes: 起始节点集合
            
        Returns:
            List[List[str]]: 检测到的所有环路列表
        """
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                # 找到环路
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
                
            if node in visited:
                return
                
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                dfs(neighbor, path.copy())
                
            rec_stack.remove(node)
        
        # 对可能在环中的每个节点运行DFS
        for node in start_nodes:
            if node not in visited:
                dfs(node, [])
                
        return cycles

    async def create_pod_from_task(self, task: Task) -> V1Pod:
        """
        从任务创建Kubernetes Pod对象
        
        Args:
            task: 任务对象
            
        Returns:
            V1Pod: Kubernetes Pod对象
        """
        # 创建资源请求
        resources = V1ResourceRequirements(
            requests={"cpu": task.cpu, "memory": task.memory},
            limits={"cpu": task.cpu, "memory": task.memory}
        )
        
        # 创建容器
        container = V1Container(
            name=task.name,
            image=task.image,
            resources=resources,
            command=task.commands,
            args=task.args,
            env=[{"name": k, "value": v} for k, v in (task.env_vars or {}).items()]
        )
        
        # 创建Pod
        pod = V1Pod(
            metadata=V1ObjectMeta(
                name=f"{task.name}-{task.task_id}",
                namespace=task.namespace,
                labels={"app": task.name, "task-id": task.task_id}
            ),
            spec=V1PodSpec(
                containers=[container],
                restart_policy="OnFailure"
            )
        )
        
        return pod

    async def deploy_task(self, task: Task) -> Tuple[bool, Optional[str]]:
        """
        部署单个任务
        
        Args:
            task: 任务对象
            
        Returns:
            Tuple[bool, Optional[str]]: 部署是否成功和错误信息
        """
        try:
            # 先检查Pod是否已经存在
            pod_name = f"{task.name}-{task.task_id}"
            try:
                existing_pod = self.core_v1.read_namespaced_pod(
                    name=pod_name,
                    namespace=task.namespace
                )
                # Pod已存在，返回成功
                logger.info(f"任务 {task.task_id} 对应的Pod已存在，跳过部署")
                return True, None
            except ApiException as e:
                # 404错误表示Pod不存在，可以继续创建
                if e.status != 404:
                    raise e
            
            # 创建新Pod
            pod = await self.create_pod_from_task(task)
            self.core_v1.create_namespaced_pod(
                namespace=task.namespace,
                body=pod
            )
            logger.info(f"成功部署任务 {task.task_id}")
            return True, None
        except ApiException as e:
            error_msg = f"部署任务 {task.task_id} 失败: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"部署任务 {task.task_id} 时发生未知错误: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    async def deploy_tasks(self, sorted_tasks: List[str], task_dict: Dict[str, Task]) -> Tuple[List[str], Dict[str, str]]:
        """
        按顺序部署任务
        
        Args:
            sorted_tasks: 排序后的任务ID列表
            task_dict: 任务字典
            
        Returns:
            Tuple[List[str], Dict[str, str]]: 成功部署的任务列表和失败的任务及原因
        """
        deployed_tasks = []
        failed_tasks = {}
        
        # 按照拓扑排序顺序逐个部署任务
        for task_id in sorted_tasks:
            task = task_dict[task_id]
            logger.info(f"开始部署任务 {task_id}（{task.name}）...")
            success, error = await self.deploy_task(task)
            
            if success:
                deployed_tasks.append(task_id)
                logger.info(f"任务 {task_id} 部署成功")
            else:
                failed_tasks[task_id] = error
                logger.error(f"任务 {task_id} 部署失败: {error}")
                # 继续部署其他任务，但记录失败情况
        
        # 确保deployed_tasks是按照拓扑排序的顺序
        logger.info(f"任务部署完成。成功: {len(deployed_tasks)}, 失败: {len(failed_tasks)}")
        return deployed_tasks, failed_tasks

    async def process_tasks_plan(self, plan: JointTasksPlan) -> TaskPlanningResponse:
        """
        处理联合任务规划
        
        Args:
            plan: 联合任务规划
            
        Returns:
            TaskPlanningResponse: 任务规划响应
        """
        try:
            # 获取修改的依赖信息（如果有）
            modified_dependencies = plan.__dict__.get('modified_dependencies', {})
            
            # 拓扑排序
            try:
                sorted_tasks, task_dict = await self.topological_sort(plan)
                logger.info(f"任务拓扑排序结果: {', '.join(sorted_tasks)}")
            except ValueError as e:
                # 拓扑排序失败（检测到环）
                return TaskPlanningResponse(
                    success=False,
                    message=str(e),
                    plan_name=plan.plan_name,
                    modified_dependencies=modified_dependencies if modified_dependencies else None
                )
            
            # 按拓扑排序顺序部署任务
            deployed_tasks, failed_tasks = await self.deploy_tasks(sorted_tasks, task_dict)
            
            # 构建响应
            success = len(failed_tasks) == 0
            
            # 构建详细的消息，包含部署顺序信息
            if success:
                message = f"所有任务成功部署，按顺序执行: {', '.join(sorted_tasks)}"
            else:
                message = f"部分任务部署失败。排序: {', '.join(sorted_tasks)}, 成功: {len(deployed_tasks)}, 失败: {len(failed_tasks)}"
            
            # 如果有修改过的依赖，添加到消息中
            if modified_dependencies:
                message += f"，{len(modified_dependencies)}个任务的依赖关系被修正"
            
            return TaskPlanningResponse(
                success=success,
                message=message,
                plan_name=plan.plan_name,
                serialized_tasks=sorted_tasks,  # 拓扑排序结果
                deployed_tasks=deployed_tasks,  # 按拓扑排序顺序部署的成功任务
                failed_tasks=failed_tasks,
                modified_dependencies=modified_dependencies if modified_dependencies else None
            )
            
        except Exception as e:
            logger.error(f"处理任务规划时发生未知错误: {str(e)}", exc_info=True)
            return TaskPlanningResponse(
                success=False,
                message=f"处理任务规划时发生未知错误: {str(e)}",
                plan_name=plan.plan_name
            ) 