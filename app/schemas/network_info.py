"""网络信息相关的数据模型"""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime


class NetworkInformation(BaseModel):
    """网络信息数据模型"""
    latency: int = Field(..., description="传输时延(ms)", ge=0)
    bandwidth: float = Field(..., description="可用带宽(Mbps)", ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "latency": 100,
                "bandwidth": 1000.5
            }
        }
    }


class DestinationNode(BaseModel):
    """目的节点信息"""
    dest_host_name: str = Field(..., description="目的节点ID", max_length=64)
    network_info: NetworkInformation = Field(..., description="网络信息")

    model_config = {
        "json_schema_extra": {
            "example": {
                "dest_host_name": "node-001",
                "network_info": {
                    "latency": 100,
                    "bandwidth": 1000.5
                }
            }
        }
    }


class NetworkReport(BaseModel):
    """网络信息上报数据模型"""
    report_id: int = Field(..., description="报告ID", ge=0, lt=2**32)
    mirror_repository_host_name: str = Field(..., description="镜像仓库节点ID", max_length=64)
    dest_host_number: int = Field(..., description="目的节点数量", ge=1, le=255)
    destinations: List[DestinationNode] = Field(..., description="目的节点列表")
    timestamp: datetime = Field(default_factory=datetime.now, description="上报时间")

    @validator('destinations')
    def validate_destinations_count(cls, v: List[DestinationNode], values: dict) -> List[DestinationNode]:
        """验证目的节点数量与dest_host_number一致"""
        if 'dest_host_number' in values and len(v) != values['dest_host_number']:
            raise ValueError("目的节点列表长度必须与dest_host_number一致")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "report_id": 12345,
                "mirror_repository_host_name": "mirror-repo-001",
                "dest_host_number": 2,
                "destinations": [
                    {
                        "dest_host_name": "node-001",
                        "network_info": {
                            "latency": 100,
                            "bandwidth": 1000.5
                        }
                    },
                    {
                        "dest_host_name": "node-002",
                        "network_info": {
                            "latency": 150,
                            "bandwidth": 800.75
                        }
                    }
                ]
            }
        }
    } 