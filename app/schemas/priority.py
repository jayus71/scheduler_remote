"""优先级相关的数据模型"""
from typing import List, Optional, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field

from app.schemas.common import Pod, Node


class NodeResource(BaseModel):
    """节点资源信息"""
    cpu: Union[str, int, float] = Field(..., description="CPU资源")
    memory: Union[str, int, float] = Field(..., description="内存资源")
    gpu: Optional[Union[str, int, float]] = Field(0, alias="nvidia.com/gpu", description="GPU资源")


class SchedulerNode(BaseModel):
    """调度节点信息"""
    name: str = Field(..., description="节点名称")
    role: Optional[str] = Field("worker", description="节点角色")
    labels: Dict[str, str] = Field(default_factory=dict, description="节点标签")
    annotations: Optional[Dict[str, str]] = Field(default_factory=dict, description="节点注解")
    capacity: NodeResource = Field(..., description="节点容量")
    allocatable: NodeResource = Field(..., description="可分配资源")
    allocated: Optional[NodeResource] = Field(None, description="已分配资源")


class HostPriority(BaseModel):
    """主机优先级信息，符合接口标识符Schedule-priority-response"""
    host: str = Field(..., description="节点主机名", alias="Host")
    score: str = Field(..., description="节点得分", alias="Score")
    
    class Config:
        populate_by_name = True
        alias_generator = lambda s: s[0].upper() + s[1:]
        arbitrary_types_allowed = True


class PriorityRequest(BaseModel):
    """优先级请求，符合接口标识符Schedule-priority-request"""
    pod: Pod = Field(..., description="Pod信息")
    nodes: List[SchedulerNode] = Field(default_factory=list, description="节点列表")
    computing_power_info: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="算力信息")
    network_info: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="网络信息")
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


class PriorityResponse(BaseModel):
    """优先级响应，符合接口标识符Schedule-priority-response"""
    hostPriorityList: List[HostPriority] = Field(
        default_factory=list,
        description="节点优先级列表"
    )
    error_msg: Optional[str] = Field("", description="错误信息", alias="error")
    
    class Config:
        populate_by_name = True
        alias_generator = lambda s: "error" if s == "error_msg" else s
        arbitrary_types_allowed = True


class RLState(BaseModel):
    """强化学习状态"""
    pod_name: str = Field(..., description="Pod名称")
    node_features: Dict[str, List[float]] = Field(..., description="节点特征矩阵")
    resource_usage: Dict[str, Dict[str, float]] = Field(..., description="资源使用情况")
    timestamp: float = Field(..., description="时间戳")


class RLAction(BaseModel):
    """强化学习动作"""
    node_weights: Dict[str, float] = Field(..., description="节点权重")


class RLExperience(BaseModel):
    """强化学习经验"""
    state: RLState = Field(..., description="状态")
    action: RLAction = Field(..., description="动作")
    reward: float = Field(..., description="奖励")
    next_state: Optional[RLState] = Field(None, description="下一个状态")
    done: bool = Field(False, description="是否完成")


class RLModelInfo(BaseModel):
    """强化学习模型信息"""
    model_name: str = Field(..., description="模型名称")
    algorithm: str = Field(..., description="算法名称")
    total_training_steps: int = Field(0, description="总训练步数")
    last_updated: float = Field(..., description="最后更新时间")
    average_reward: float = Field(0.0, description="平均奖励")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="超参数")
