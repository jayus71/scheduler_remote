"""网络信息上报数据模型"""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class NetworkInfo(BaseModel):
    """网络信息结构"""
    latency: int = Field(..., description="传输时延(ms)")
    bandwidth_integer: int = Field(..., description="可用带宽整数部分(Mbps)")
    bandwidth_decimal: int = Field(..., description="可用带宽小数部分(Mbps)")
    
    @property
    def bandwidth(self) -> float:
        """返回完整的带宽值"""
        return self.bandwidth_integer + self.bandwidth_decimal / 100.0


class DestinationHost(BaseModel):
    """目的节点信息"""
    host_name: str = Field(..., description="目的节点ID")
    network_info: NetworkInfo = Field(..., description="网络信息")


class NetworkReport(BaseModel):
    """网络信息上报数据模型"""
    report_id: int = Field(..., description="报告ID")
    mirror_repository_host_name: str = Field(..., description="镜像仓库节点ID")
    destinations: List[DestinationHost] = Field(default_factory=list, description="目的节点列表")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "report_id": 12345,
                "mirror_repository_host_name": "repository-node-1",
                "destinations": [
                    {
                        "host_name": "node-1",
                        "network_info": {
                            "latency": 10,
                            "bandwidth_integer": 100,
                            "bandwidth_decimal": 50
                        }
                    }
                ]
            }
        }
    }


class NetworkInfoStore(BaseModel):
    """网络信息存储"""
    reports: Dict[str, Dict[str, NetworkInfo]] = Field(
        default_factory=dict, 
        description="存储从源节点到目标节点的网络信息，格式为: {source_node: {dest_node: network_info}}"
    )
    
    def update_network_info(self, report: NetworkReport):
        """更新网络信息存储
        
        Args:
            report: 网络信息上报
        """
        source = report.mirror_repository_host_name
        
        if source not in self.reports:
            self.reports[source] = {}
            
        for dest in report.destinations:
            self.reports[source][dest.host_name] = dest.network_info
    
    def get_network_info(self, source: str, dest: str) -> Optional[NetworkInfo]:
        """获取从源节点到目标节点的网络信息
        
        Args:
            source: 源节点
            dest: 目标节点
            
        Returns:
            NetworkInfo: 网络信息，如果不存在则返回None
        """
        if source in self.reports and dest in self.reports[source]:
            return self.reports[source][dest]
        return None 