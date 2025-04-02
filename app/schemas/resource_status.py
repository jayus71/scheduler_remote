from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, validator


class ResourceStatus(str, Enum):
    """资源状态枚举"""
    AVAILABLE = 'available'
    IN_USE = 'in_use'
    ERROR = 'error'


class ResourceStatusRequest(BaseModel):
    """资源状态请求模型
    
    Attributes:
        request_id: 请求的唯一标识符
        request_time: 请求发送的时间戳
        error_message: 如果请求失败，记录相关错误信息
    """
    request_id: str = Field(
        ...,
        description="请求的唯一标识符",
        example="req-123456"
    )
    request_time: datetime = Field(
        default_factory=datetime.now,
        description="请求发送的时间戳",
        example="2024-03-12T10:30:00"
    )
    error_message: Optional[str] = Field(
        None,
        description="如果请求失败，记录相关错误信息",
        example="节点未找到"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "req-123456",
                "request_time": "2024-03-12T10:30:00",
                "error_message": None
            }
        }


class ResourceStatusInfo(BaseModel):
    """资源状态信息模型
    
    Attributes:
        node_id: 平台资源中节点的唯一标识符
        status_id: 状态记录的唯一标识符
        timestamp: 记录资源状态的时间
        cpu_status: 当前CPU资源的状态
        cpu_usage: CPU资源的使用百分比
        gpu_status: 当前GPU资源的状态
        gpu_usage: GPU资源的使用百分比
        memory_status: 当前内存资源的状态
        memory_usage: 内存资源的使用百分比
        disk_status: 当前磁盘资源的状态
        disk_usage: 磁盘资源的使用百分比
        network_status: 当前网络资源的状态
        network_usage: 网络带宽的使用百分比
        overall_load: 系统综合负载使用百分比
        response_timestamp: 资源信息响应返回的时间戳
    """
    # 节点信息
    node_id: str = Field(
        ...,
        description="平台资源中节点的唯一标识符",
        example="node-001"
    )
    status_id: str = Field(
        ...,
        description="状态记录的唯一标识符",
        example="status-001"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="记录资源状态的时间"
    )
    
    # CPU相关
    cpu_status: ResourceStatus = Field(
        ...,
        description="当前CPU资源的状态"
    )
    cpu_usage: float = Field(
        ...,
        ge=0,
        le=100,
        description="CPU资源的使用百分比"
    )
    
    # GPU相关
    gpu_status: ResourceStatus = Field(
        ...,
        description="当前GPU资源的状态"
    )
    gpu_usage: float = Field(
        ...,
        ge=0,
        le=100,
        description="GPU资源的使用百分比"
    )
    
    # 内存相关
    memory_status: ResourceStatus = Field(
        ...,
        description="当前内存资源的状态"
    )
    memory_usage: float = Field(
        ...,
        ge=0,
        le=100,
        description="内存资源的使用百分比"
    )
    
    # 磁盘相关
    disk_status: ResourceStatus = Field(
        ...,
        description="当前磁盘资源的状态"
    )
    disk_usage: float = Field(
        ...,
        ge=0,
        le=100,
        description="磁盘资源的使用百分比"
    )
    
    # 网络相关
    network_status: ResourceStatus = Field(
        ...,
        description="当前网络资源的状态"
    )
    network_usage: float = Field(
        ...,
        ge=0,
        le=100,
        description="网络带宽的使用百分比"
    )
    
    # 综合信息
    overall_load: float = Field(
        ...,
        ge=0,
        le=100,
        description="系统综合负载使用百分比"
    )
    response_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="资源信息响应返回的时间戳"
    )

    @validator('cpu_usage', 'gpu_usage', 'memory_usage', 'disk_usage', 'network_usage', 'overall_load')
    def validate_usage(cls, v: float) -> float:
        """验证使用率是否在有效范围内"""
        if not 0 <= v <= 100:
            raise ValueError("使用率必须在0-100之间")
        return round(v, 2)  # 保留两位小数

    class Config:
        json_schema_extra = {
            "example": {
                "node_id": "node-001",
                "status_id": "status-001",
                "timestamp": "2024-03-12T10:30:00",
                "cpu_status": "available",
                "cpu_usage": 45.5,
                "gpu_status": "in_use",
                "gpu_usage": 78.3,
                "memory_status": "available",
                "memory_usage": 60.0,
                "disk_status": "available",
                "disk_usage": 55.0,
                "network_status": "available",
                "network_usage": 30.0,
                "overall_load": 58.2,
                "response_timestamp": "2024-03-12T10:30:01"
            }
        } 