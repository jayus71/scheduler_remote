"""Pod管理服务"""
from typing import Dict

from app.schemas.pods import Pod
from app.services.k8s_service import KubernetesService


class PodsService:
    """Pod管理服务类"""

    def __init__(self):
        """初始化Pod管理服务"""
        self.k8s_service = KubernetesService()

    def create_pod(self, namespace: str, pod: Dict) -> Dict:
        """创建Pod
        
        Args:
            namespace: 命名空间
            pod: Pod配置
            
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
