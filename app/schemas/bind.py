"""节点绑定模型模块"""
from typing import Optional
from pydantic import BaseModel, Field


class BindRequest(BaseModel):
    """
    节点绑定请求模型
    
    用于将Pod绑定到特定节点上的请求
    
    接口标识符: Schedule-Bind-request
    """
    pod_name: str = Field(..., description="容器名", alias="PodName")
    pod_namespace: str = Field(..., description="容器命名空间", alias="PodNamespace")
    pod_uid: str = Field(..., description="容器标识", alias="PodUID")
    node: str = Field(..., description="节点名", alias="Node")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "PodName": "example-pod",
                "PodNamespace": "default",
                "PodUID": "12345678-1234-1234-1234-123456789012",
                "Node": "worker-node-1"
            }
        }
    }


class BindResponse(BaseModel):
    """
    节点绑定响应模型
    
    节点绑定操作的响应结果
    
    接口标识符: Schedule-Bind-response
    """
    error: Optional[str] = Field(None, description="错误信息，如果为None则表示绑定成功")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": None
            }
        }
    }
