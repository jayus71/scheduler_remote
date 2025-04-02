"""节点过滤数据模型"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas.common import Pod, Node, NodeScore


class FilterResult(BaseModel):
    """过滤结果"""
    nodes: List[Node] = Field(default_factory=list)
    node_names: List[str] = Field(default_factory=list)
    failed_nodes: Dict[str, str] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {
                "nodes": [],
                "node_names": [],
                "failed_nodes": {}
            }
        }
    }


class FilterRequest(BaseModel):
    """过滤请求"""
    pod: Pod
    nodes: List[Node]

    model_config = {
        "json_schema_extra": {
            "example": {
                "pod": {},
                "nodes": []
            }
        }
    }


class FilterResponse(BaseModel):
    """过滤响应"""
    nodes: List[Node] = Field(default_factory=list)
    node_names: List[str] = Field(default_factory=list)
    failed_nodes: Dict[str, str] = Field(default_factory=dict)
    error: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "nodes": [],
                "node_names": [],
                "failed_nodes": {},
                "error": None
            }
        }
    }
