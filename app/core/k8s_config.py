"""Kubernetes配置模块"""
import os
from typing import Tuple, Optional
from kubernetes import client, config
from loguru import logger



def load_kubernetes_config():
    """
    加载Kubernetes配置
    
    按以下顺序尝试加载配置：
    1. 如果设置了USE_SERVICE_ACCOUNT=true，优先使用ServiceAccount配置
    2. 如果ServiceAccount配置失败或未启用，尝试使用本地kubeconfig配置
    """
    # 延迟导入settings，避免循环导入
    from app.core.config import settings
    

    use_service_account = settings.USE_SERVICE_ACCOUNT
    
    logger.info(f"加载Kubernetes配置: USE_SERVICE_ACCOUNT={use_service_account}")
    
    try:
        # 如果设置了service account，优先使用ServiceAccount配置
        if use_service_account:
            try:
                # 使用集群内ServiceAccount配置
                config.load_incluster_config()
                logger.info("已使用ServiceAccount加载Kubernetes配置")
                
                # 验证配置是否有效
                v1 = client.CoreV1Api()
                try:
                    # 尝试列出命名空间，验证连接
                    v1.list_namespace(limit=1)
                    logger.info("Kubernetes API连接验证成功")
                except Exception as e:
                    logger.warning(f"Kubernetes API连接验证失败: {str(e)}")
                    raise Exception("ServiceAccount配置无效")
                
                return
            except Exception as e:
                logger.warning(f"使用ServiceAccount加载配置失败，将尝试本地配置: {str(e)}")
        
        # 尝试加载本地配置
        try:
            # 首先尝试从项目根目录加载kubeconfig.yaml
            kubeconfig_path = os.path.join(os.getcwd(), "kubeconfig.yaml")
            logger.info(f"尝试加载kubeconfig文件: {kubeconfig_path}")
            
            if os.path.exists(kubeconfig_path):
                # 检查文件大小，确保不是空文件
                if os.path.getsize(kubeconfig_path) > 0:
                    logger.info(f"找到kubeconfig文件: {kubeconfig_path}, 大小: {os.path.getsize(kubeconfig_path)} 字节")
                    config.load_kube_config(kubeconfig_path)
                    logger.info(f"已加载本地Kubernetes配置: {kubeconfig_path}")
                else:
                    logger.warning(f"kubeconfig文件存在但为空: {kubeconfig_path}")
                    raise FileNotFoundError(f"kubeconfig文件存在但为空: {kubeconfig_path}")
            else:
                logger.warning(f"未找到kubeconfig文件: {kubeconfig_path}")
                
                # 如果项目根目录没有kubeconfig.yaml，尝试默认路径
                default_kubeconfig = os.path.expanduser("~/.kube/config")
                logger.info(f"尝试加载默认kubeconfig文件: {default_kubeconfig}")
                
                if os.path.exists(default_kubeconfig):
                    logger.info(f"找到默认kubeconfig文件: {default_kubeconfig}")
                    config.load_kube_config(default_kubeconfig)
                    logger.info(f"已加载默认Kubernetes配置: {default_kubeconfig}")
                else:
                    logger.warning(f"未找到默认kubeconfig文件: {default_kubeconfig}")
                    
                    # 尝试从环境变量KUBECONFIG指定的路径加载
                    kubeconfig_env = os.environ.get("KUBECONFIG")
                    if kubeconfig_env:
                        logger.info(f"尝试从环境变量加载kubeconfig文件: {kubeconfig_env}")
                        if os.path.exists(kubeconfig_env):
                            logger.info(f"找到环境变量指定的kubeconfig文件: {kubeconfig_env}")
                            config.load_kube_config(kubeconfig_env)
                            logger.info(f"已加载环境变量指定的Kubernetes配置: {kubeconfig_env}")
                        else:
                            logger.warning(f"环境变量指定的kubeconfig文件不存在: {kubeconfig_env}")
                            raise FileNotFoundError(f"环境变量指定的kubeconfig文件不存在: {kubeconfig_env}")
                    else:
                        logger.warning("未设置KUBECONFIG环境变量")
                        raise FileNotFoundError("未找到有效的kubeconfig文件，请确保kubeconfig.yaml文件存在且不为空")
            
            # 验证配置是否有效
            v1 = client.CoreV1Api()
            try:
                # 尝试列出命名空间，验证连接
                v1.list_namespace(limit=1)
                logger.info("Kubernetes API连接验证成功")
            except Exception as e:
                logger.warning(f"Kubernetes API连接验证失败: {str(e)}")
                raise Exception(f"本地kubeconfig配置无效: {str(e)}")
                
            return
        except Exception as e:
            logger.error(f"无法加载本地Kubernetes配置: {str(e)}")
            
                
            raise Exception(f"无法加载本地Kubernetes配置: {str(e)}")
            
    except Exception as e:
        logger.error(f"Kubernetes配置加载失败: {str(e)}")
        raise


def create_k8s_client_from_config(
    api_server: str, 
    token_path: str, 
    cert_path: str
) -> Tuple[client.CoreV1Api, client.CustomObjectsApi]:
    """
    使用指定的配置创建Kubernetes客户端
    
    Args:
        api_server: Kubernetes API服务器地址
        token_path: ServiceAccount令牌路径
        cert_path: ServiceAccount证书路径
        
    Returns:
        Tuple[CoreV1Api, CustomObjectsApi]: Kubernetes API客户端
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(token_path):
            raise FileNotFoundError(f"令牌文件不存在: {token_path}")
        if not os.path.exists(cert_path):
            raise FileNotFoundError(f"证书文件不存在: {cert_path}")
        
        # 读取令牌
        with open(token_path, 'r') as token_file:
            token = token_file.read().strip()
        
        # 配置API客户端
        configuration = client.Configuration()
        configuration.host = api_server
        configuration.api_key['authorization'] = f'Bearer {token}'
        configuration.ssl_ca_cert = cert_path
        
        # 创建API客户端上下文
        api_client = client.ApiClient(configuration)
        
        # 创建API客户端
        core_v1 = client.CoreV1Api(api_client)
        custom_objects = client.CustomObjectsApi(api_client)
        
        logger.info(f"已创建Kubernetes客户端，连接到: {api_server}")
        return core_v1, custom_objects
        
    except Exception as e:
        logger.error(f"创建Kubernetes客户端失败: {str(e)}")
        raise


def get_k8s_clients() -> Tuple[client.CoreV1Api, client.CustomObjectsApi]:
    """
    获取Kubernetes客户端
    
    Returns:
        Tuple[CoreV1Api, CustomObjectsApi]: Kubernetes API客户端
    """
    
    # 加载配置
    load_kubernetes_config()

    # 创建API客户端
    core_v1 = client.CoreV1Api()
    custom_objects = client.CustomObjectsApi()

    return core_v1, custom_objects


def get_external_k8s_clients(
    api_server: Optional[str] = None,
    token_path: Optional[str] = None,
    cert_path: Optional[str] = None
) -> Tuple[client.CoreV1Api, client.CustomObjectsApi]:
    """
    获取外部Kubernetes集群的客户端
    
    Args:
        api_server: Kubernetes API服务器地址，默认使用settings.EXTERNAL_K8S_HOST
        token_path: ServiceAccount令牌路径，默认使用settings.EXTERNAL_K8S_TOKEN_PATH
        cert_path: ServiceAccount证书路径，默认使用settings.EXTERNAL_K8S_CERT_PATH
        
    Returns:
        Tuple[CoreV1Api, CustomObjectsApi]: Kubernetes API客户端
    """
    # 延迟导入settings，避免循环导入
    from app.core.config import settings
    
    
    # 使用默认值或传入的参数
    api_server = api_server or settings.EXTERNAL_K8S_HOST
    token_path = token_path or settings.EXTERNAL_K8S_TOKEN_PATH
    cert_path = cert_path or settings.EXTERNAL_K8S_CERT_PATH
    
    logger.info(f"连接到外部Kubernetes集群: {api_server}")
    
    return create_k8s_client_from_config(
        api_server=api_server,
        token_path=token_path,
        cert_path=cert_path
    )


# 导出函数
__all__ = [
    'load_kubernetes_config',
    'create_k8s_client_from_config',
    'get_k8s_clients',
    'get_external_k8s_clients'
]
