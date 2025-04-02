"""节点绑定服务模块"""
from typing import Dict, Optional
from kubernetes.client import V1Binding, V1ObjectReference, V1ObjectMeta, ApiException
from loguru import logger

from app.core.k8s_config import get_k8s_clients
from app.schemas.bind import BindRequest


class BindService:
    """节点绑定服务类"""

    def __init__(self):
        """初始化Kubernetes客户端"""
        self.core_v1, _ = get_k8s_clients()

    async def bind_pod_to_node(self, request: BindRequest) -> Optional[str]:
        """
        将Pod绑定到指定节点
        
        Args:
            request: 绑定请求，包含Pod和目标节点信息
            
        Returns:
            Optional[str]: 如果绑定失败，返回错误信息；如果成功，返回None
        """
        try:
            logger.info(f"开始将Pod {request.pod_name} 绑定到节点 {request.node}")

            # 检查Pod是否已经被分配
            pod = self.core_v1.read_namespaced_pod(
                name=request.pod_name,
                namespace=request.pod_namespace
            )
            
            if pod.spec.node_name:
                return f"Pod {request.pod_name} 已经被分配到节点 {pod.spec.node_name}"

            # 创建绑定对象
            target = V1ObjectReference(
                api_version="v1",
                kind="Node",
                name=request.node
            )

            binding = V1Binding(
                metadata=V1ObjectMeta(
                    name=request.pod_name,
                    namespace=request.pod_namespace,
                    uid=request.pod_uid
                ),
                target=target
            )

            # 执行绑定操作
            self.core_v1.create_namespaced_binding(
                namespace=request.pod_namespace,
                body=binding
            )

            logger.info(f"Pod {request.pod_name} 成功绑定到节点 {request.node}")
            return None

        except ApiException as e:
            error_msg = f"绑定Pod到节点失败: {str(e)}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"绑定过程中发生未知错误: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
