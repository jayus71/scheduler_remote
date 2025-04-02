"""路由信息查询接口数据模型"""
from typing import List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field


class RouteInfoRequest(BaseModel):
    """路由信息请求表模型
    
    接口标识符: Route-Info-Query
    """
    request_id: str = Field(..., description="资源请求唯一标识符")
    request_time: datetime = Field(default_factory=datetime.now, description="请求时间戳")
    source_nodes: Optional[List[str]] = Field(None, description="源节点ID列表（可选，若不指定则传输全部节点）")
    target_nodes: Optional[List[str]] = Field(None, description="目标节点ID列表（可选，若不指定则传输全部节点）")


class RouteStatusResponse(BaseModel):
    """路由状态响应表模型"""
    source_node: str = Field(..., description="源节点的标识符")
    target_node: str = Field(..., description="目标节点的标识符")
    connection_status: Literal["direct", "indirect", "unavailable"] = Field(..., description="节点间的连接状态")
    latency: float = Field(..., gt=0, description="网络延迟(ms)")
    bandwidth: float = Field(..., gt=0, description="可用带宽(Mbps)")
    route_path: List[str] = Field(..., description="完整路由路径的节点列表")
    response_timestamp: datetime = Field(default_factory=datetime.now, description="路由信息响应返回的时间戳")


class RouteInfoResponse(BaseModel):
    """路由信息响应聚合模型"""
    request_id: str = Field(..., description="对应请求的唯一标识符")
    routes: List[RouteStatusResponse] = Field(..., description="路由状态响应列表") 