"""通用数据模型"""
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class ContainerResources(BaseModel):
    """容器资源需求和限制"""
    cpu: str = Field(..., description="CPU资源")
    memory: str = Field(..., description="内存资源")
    gpu: Optional[str] = Field(None, alias="nvidia.com/gpu", description="GPU资源")

    model_config = {
        "json_schema_extra": {
            "example": {
                "cpu": "100m",
                "memory": "128Mi",
                "nvidia.com/gpu": "1"
            }
        }
    }


class ContainerResourceRequirements(BaseModel):
    """容器资源配置"""
    requests: ContainerResources = Field(..., description="资源请求")
    limits: Optional[ContainerResources] = Field(None, description="资源限制")

    model_config = {
        "json_schema_extra": {
            "example": {
                "requests": {
                    "cpu": "100m",
                    "memory": "128Mi"
                },
                "limits": {
                    "cpu": "200m",
                    "memory": "256Mi"
                }
            }
        }
    }


class Container(BaseModel):
    """容器规格"""
    name: str
    image: Optional[str] = None
    resources: Optional[ContainerResourceRequirements] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "container-1",
                "image": "nginx:latest",
                "resources": {
                    "requests": {
                        "cpu": "100m",
                        "memory": "128Mi"
                    },
                    "limits": {
                        "cpu": "200m",
                        "memory": "256Mi"
                    }
                }
            }
        }
    }


class PodMetadata(BaseModel):
    """Pod元数据"""
    name: str
    namespace: Optional[str] = "default"
    uid: Optional[str] = None
    labels: Optional[Dict[str, str]] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "example-pod",
                "namespace": "default",
                "labels": {
                    "app": "example"
                }
            }
        }
    }


class PodSpec(BaseModel):
    """Pod规格"""
    containers: List[Container]
    node_selector: Optional[Dict[str, str]] = None
    tolerations: Optional[List[Dict[str, Any]]] = None
    nodeName: Optional[str] = Field(None, description="指定容器被调度到的节点")

    model_config = {
        "json_schema_extra": {
            "example": {
                "containers": [
                    {
                        "name": "container-1",
                        "image": "nginx:latest",
                        "resources": {
                            "requests": {
                                "cpu": "100m",
                                "memory": "128Mi"
                            },
                            "limits": {
                                "cpu": "200m",
                                "memory": "256Mi"
                            }
                        }
                    }
                ],
                "node_selector": {
                    "kubernetes.io/os": "linux"
                },
                "nodeName": "node-1"
            }
        }
    }


class Pod(BaseModel):
    """Pod信息"""
    metadata: PodMetadata
    spec: PodSpec

    model_config = {
        "json_schema_extra": {
            "example": {
                "metadata": {
                    "name": "example-pod",
                    "namespace": "default"
                },
                "spec": {
                    "containers": [
                        {
                            "name": "container-1",
                            "image": "nginx:latest",
                            "resources": {
                                "requests": {
                                    "cpu": "100m",
                                    "memory": "128Mi"
                                },
                                "limits": {
                                    "cpu": "200m",
                                    "memory": "256Mi"
                                }
                            }
                        }
                    ]
                }
            }
        }
    }


class NodeCapacity(BaseModel):
    """节点容量"""
    cpu: Union[str, int, float]
    memory: Union[str, int, float]
    pods: Optional[Union[str, int]] = None
    gpu: Optional[int] = Field(0, alias="nvidia.com/gpu", description="GPU卡数量")

    model_config = {
        "json_schema_extra": {
            "example": {
                "cpu": "4",
                "memory": "8Gi",
                "pods": "110",
                "gpu": 2
            }
        }
    }


class NodeAddress(BaseModel):
    """节点地址信息"""
    internalIP: str = Field(..., description="内部IP地址")
    hostname: str = Field(..., description="主机名")

    model_config = {
        "json_schema_extra": {
            "example": {
                "internalIP": "192.168.1.100",
                "hostname": "node-1"
            }
        }
    }


class Node(BaseModel):
    """节点信息"""
    name: str
    role: Optional[str] = "worker"
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Optional[Dict[str, str]] = Field(default_factory=dict)
    addresses: Optional[NodeAddress] = None
    internalIP: Optional[str] = None
    hostname: Optional[str] = None
    capacity: NodeCapacity
    allocatable: Optional[NodeCapacity] = None
    cpuUsage: Optional[float] = None
    memoryUsage: Optional[float] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "node-1",
                "role": "worker",
                "labels": {
                    "kubernetes.io/os": "linux",
                    "kubernetes.io/arch": "amd64"
                },
                "internalIP": "192.168.1.100",
                "hostname": "node-1",
                "capacity": {
                    "cpu": "4",
                    "memory": "8Gi",
                    "pods": "110"
                },
                "allocatable": {
                    "cpu": "3.8",
                    "memory": "7.5Gi",
                    "pods": "110"
                },
                "cpuUsage": 0.45,
                "memoryUsage": 0.65
            }
        }
    }


class NodeScore(BaseModel):
    """节点得分"""
    name: str
    score: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "node-1",
                "score": 85
            }
        }
    }


class PodStatus(BaseModel):
    """Pod状态信息"""
    phase: str = Field(..., description="Pod阶段")
    conditions: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Pod状态条件")
    message: Optional[str] = Field(None, description="状态信息")
    reason: Optional[str] = Field(None, description="状态原因")
    host_ip: Optional[str] = Field(None, description="主机IP")
    pod_ip: Optional[str] = Field(None, description="Pod IP")
    start_time: Optional[str] = Field(None, description="启动时间")
    container_statuses: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="容器状态列表")

    model_config = {
        "json_schema_extra": {
            "example": {
                "phase": "Running",
                "conditions": [
                    {
                        "type": "Ready",
                        "status": "True",
                        "lastProbeTime": None,
                        "lastTransitionTime": "2024-03-14T10:00:00Z"
                    }
                ],
                "message": None,
                "reason": None,
                "host_ip": "192.168.1.100",
                "pod_ip": "10.244.0.15",
                "start_time": "2024-03-14T10:00:00Z",
                "container_statuses": [
                    {
                        "name": "nginx",
                        "state": {"running": {"startedAt": "2024-03-14T10:00:00Z"}},
                        "ready": True,
                        "restartCount": 0,
                        "image": "nginx:latest",
                        "imageID": "docker-pullable://nginx@sha256:..."
                    }
                ]
            }
        }
    }
