"""Kubernetes服务模块"""
from typing import Dict, List, Optional

from kubernetes.client import (
    V1Pod,
    ApiException,
    CoreV1Api,
    CustomObjectsApi
)
from loguru import logger

from app.core.k8s_config import get_k8s_clients
from app.schemas.common import Node, NodeAddress


class KubernetesService:
    """Kubernetes服务类"""

    def __init__(self):
        """初始化Kubernetes客户端"""
        self.core_v1, self.custom_objects = get_k8s_clients()

    def list_nodes(self) -> List[Node]:
        """
        获取所有节点信息
        
        Returns:
            List[Node]: 节点列表
        """
        try:
            nodes = self.core_v1.list_node()
            result = []

            for node in nodes.items:
                # 获取节点基本信息
                node_info = Node(
                    name=node.metadata.name,
                    role=node.metadata.labels.get("node-role.kubernetes.io/worker", "worker"),
                    labels=node.metadata.labels or {},
                    annotations=node.metadata.annotations or {},
                    addresses=NodeAddress(internalIP="", hostname=node.metadata.name),
                    capacity={
                        "cpu": int(node.status.capacity["cpu"]),
                        "memory": self._convert_memory_to_mb(node.status.capacity["memory"]),
                        "pods": int(node.status.capacity.get("pods", 0)),
                        "nvidia.com/gpu": int(node.status.capacity.get("nvidia.com/gpu", 0))
                    },
                    allocatable={
                        "cpu": int(node.status.allocatable["cpu"]),
                        "memory": self._convert_memory_to_mb(node.status.allocatable["memory"]),
                        "pods": int(node.status.allocatable.get("pods", 0)),
                        "nvidia.com/gpu": int(node.status.allocatable.get("nvidia.com/gpu", 0))
                    }
                )

                # 获取节点IP地址
                for address in node.status.addresses:
                    if address.type == "InternalIP":
                        node_info.addresses.internalIP = address.address
                    elif address.type == "Hostname":
                        node_info.addresses.hostname = address.address

                # 获取节点资源使用情况
                try:
                    node_info.cpuUsage = self._get_node_cpu_usage(node.metadata.name)
                    node_info.memoryUsage = self._get_node_memory_usage(node.metadata.name)
                except Exception as e:
                    logger.warning(f"获取节点 {node.metadata.name} 资源使用情况失败: {str(e)}")

                result.append(node_info)

            return result
        except Exception as e:
            logger.error(f"获取节点列表失败: {str(e)}")
            return []

    def get_node(self, node_name: str) -> Optional[Node]:
        """
        根据节点名称获取节点信息
        
        Args:
            node_name: 节点名称
            
        Returns:
            Optional[Node]: 节点信息，如果找不到则返回None
        """
        try:
            node = self.core_v1.read_node(name=node_name)
            
            # 创建节点信息对象
            node_info = Node(
                name=node.metadata.name,
                role=node.metadata.labels.get("node-role.kubernetes.io/worker", "worker"),
                labels=node.metadata.labels or {},
                annotations=node.metadata.annotations or {},
                addresses=NodeAddress(internalIP="", hostname=node.metadata.name),
                capacity={
                    "cpu": int(node.status.capacity["cpu"]),
                    "memory": self._convert_memory_to_mb(node.status.capacity["memory"]),
                    "pods": int(node.status.capacity.get("pods", 0)),
                    "nvidia.com/gpu": int(node.status.capacity.get("nvidia.com/gpu", 0))
                },
                allocatable={
                    "cpu": int(node.status.allocatable["cpu"]),
                    "memory": self._convert_memory_to_mb(node.status.allocatable["memory"]),
                    "pods": int(node.status.allocatable.get("pods", 0)),
                    "nvidia.com/gpu": int(node.status.allocatable.get("nvidia.com/gpu", 0))
                }
            )
            
            # 获取节点IP地址
            for address in node.status.addresses:
                if address.type == "InternalIP":
                    node_info.addresses.internalIP = address.address
                elif address.type == "Hostname":
                    node_info.addresses.hostname = address.address
                    
            # 获取节点资源使用情况
            try:
                node_info.cpuUsage = self._get_node_cpu_usage(node.metadata.name)
                node_info.memoryUsage = self._get_node_memory_usage(node.metadata.name)
            except Exception as e:
                logger.warning(f"获取节点 {node.metadata.name} 资源使用情况失败: {str(e)}")
                
            return node_info
            
        except ApiException as e:
            if e.status == 404:
                logger.warning(f"节点 {node_name} 不存在")
                return None
            else:
                logger.error(f"获取节点 {node_name} 信息失败: {str(e)}")
                return None
        except Exception as e:
            logger.error(f"获取节点 {node_name} 信息失败: {str(e)}")
            return None

    def _convert_memory_to_mb(self, memory: str) -> int:
        """
        将Kubernetes内存单位转换为MB
        
        Args:
            memory: 内存大小字符串（如 "1Gi"）
            
        Returns:
            int: MB为单位的内存大小
        """
        if memory.endswith("Ki"):
            return int(memory[:-2]) // 1024
        elif memory.endswith("Mi"):
            return int(memory[:-2])
        elif memory.endswith("Gi"):
            return int(memory[:-2]) * 1024
        else:
            return int(memory) // (1024 * 1024)  # 假设为字节

    def _get_node_metrics(self, node_name: str, metric_type: str) -> int:
        """
        获取节点指标数据
        
        Args:
            node_name: 节点名称
            metric_type: 指标类型，'cpu'或'memory'
            
        Returns:
            int: 指标值（CPU为毫核，内存为MB）
        """
        try:
            metrics = self.custom_objects.list_cluster_custom_object(
                group="metrics.k8s.io",
                version="v1beta1",
                plural="nodes"
            )

            for item in metrics.get("items", []):
                if item.get("metadata", {}).get("name") == node_name:
                    usage = item.get("usage", {}).get(metric_type, "0")

                    if metric_type == "cpu":
                        # 将CPU使用量转换为毫核
                        if usage.endswith("n"):
                            return int(usage[:-1]) // 1000000
                        elif usage.endswith("u"):
                            return int(usage[:-1]) // 1000
                        elif usage.endswith("m"):
                            return int(usage[:-1])
                        else:
                            return int(usage) * 1000
                    else:  # memory
                        return self._convert_memory_to_mb(usage)
            return 0
        except ApiException as e:
            logger.error(f"获取节点{metric_type}使用量失败: {e}")
            return 0

    def _get_node_cpu_usage(self, node_name: str) -> int:
        """
        获取节点CPU使用量（单位：毫核）
        
        Args:
            node_name: 节点名称
            
        Returns:
            int: CPU使用量（毫核）
        """
        return self._get_node_metrics(node_name, "cpu")

    def _get_node_memory_usage(self, node_name: str) -> int:
        """
        获取节点内存使用量（单位：MB）
        
        Args:
            node_name: 节点名称
            
        Returns:
            int: 内存使用量（MB）
        """
        return self._get_node_metrics(node_name, "memory")

    def create_pod(self, namespace: str, pod: V1Pod) -> Optional[V1Pod]:
        """
        在指定命名空间中创建Pod
        
        Args:
            namespace: 命名空间
            pod: Pod对象
            
        Returns:
            Optional[V1Pod]: 创建的Pod对象,如果创建失败则返回None
        """
        try:
            created_pod = self.core_v1.create_namespaced_pod(
                namespace=namespace,
                body=pod
            )
            logger.info(f"Pod {pod.metadata.name} 创建成功")
            return created_pod
        except ApiException as e:
            logger.error(f"创建Pod失败: {e}")
            return None

    def delete_pod(self, namespace: str, pod_name: str):
        """
        删除Pod
        
        Args:
            namespace: 命名空间
            pod_name: Pod名称
        """
        try:
            self.core_v1.delete_namespaced_pod(
                name=pod_name,
                namespace=namespace
            )
            logger.info(f"Pod {pod_name} 删除成功")
        except ApiException as e:
            logger.error(f"删除Pod失败: {str(e)}")
            raise

    def get_pod_status(self, namespace: str, pod_name: str) -> Dict:
        """
        获取Pod状态
        
        Args:
            namespace: 命名空间
            pod_name: Pod名称
            
        Returns:
            Dict: Pod状态信息
            
        Raises:
            ApiException: 当API调用失败时
        """
        try:
            logger.info(f"正在获取Pod状态: namespace={namespace}, pod_name={pod_name}")

            pod = self.core_v1.read_namespaced_pod(
                name=pod_name,
                namespace=namespace
            )

            status = {
                "name": pod.metadata.name,
                "namespace": pod.metadata.namespace,
                "phase": pod.status.phase,
                "hostIP": pod.status.host_ip,
                "podIP": pod.status.pod_ip,
                "startTime": pod.status.start_time,
                "conditions": [
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "lastTransitionTime": condition.last_transition_time
                    }
                    for condition in (pod.status.conditions or [])
                ]
            }

            logger.info(f"成功获取Pod状态: {status}")
            return status

        except ApiException as e:
            if e.status == 404:
                logger.warning(f"Pod不存在: namespace={namespace}, pod_name={pod_name}")
            else:
                logger.error(f"获取Pod状态失败: {str(e)}")
            raise
