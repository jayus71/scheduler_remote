"""路由信息服务

处理路由信息查询接口（Route-Info-Query）的业务逻辑
"""
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from loguru import logger

from app.schemas.route_info import RouteInfoRequest, RouteStatusResponse, RouteInfoResponse
from app.services.network_info_service import NetworkInfoService
from app.core.app_state import get_network_info_service


class RouteInfoService:
    """路由信息查询服务类"""

    def __init__(self):
        """初始化路由信息服务"""
        self.logger = logger

    async def get_route_info(self, request: RouteInfoRequest) -> RouteInfoResponse:
        """获取路由信息
        
        根据请求参数查询路由信息
        
        Args:
            request: 路由信息请求
            
        Returns:
            RouteInfoResponse: 路由信息响应
        """
        self.logger.info(f"处理路由信息查询请求: {request.request_id}")
        
        # 获取网络信息服务实例
        network_info_service = get_network_info_service()
        if not network_info_service:
            self.logger.error("网络信息服务未初始化")
            return RouteInfoResponse(
                request_id=request.request_id,
                routes=[]
            )
        
        # 获取源节点和目标节点列表
        source_nodes = request.source_nodes or await self._get_all_nodes(network_info_service)
        target_nodes = request.target_nodes or await self._get_all_nodes(network_info_service)
        
        # 查询所有路由信息
        routes: List[RouteStatusResponse] = []
        for source in source_nodes:
            for target in target_nodes:
                if source == target:
                    continue  # 跳过相同节点
                
                # 查询网络信息
                route_info = await self._query_route_info(network_info_service, source, target)
                if route_info:
                    routes.append(route_info)
        
        self.logger.info(f"路由信息查询完成, 返回 {len(routes)} 条路由信息")
        return RouteInfoResponse(
            request_id=request.request_id,
            routes=routes
        )
    
    async def _get_all_nodes(self, network_info_service: NetworkInfoService) -> List[str]:
        """获取所有节点
        
        当请求中未指定源节点或目标节点时，获取所有可用节点
        
        Args:
            network_info_service: 网络信息服务实例
            
        Returns:
            List[str]: 所有节点ID列表
        """
        # 从最新的网络报告中提取节点信息
        all_nodes = set()
        for report_key, report in network_info_service.latest_reports.items():
            # 添加镜像仓库节点
            all_nodes.add(report_key)
            
            # 添加目标节点
            for dest in report.destinations:
                all_nodes.add(dest.host_name)
        
        return list(all_nodes)
    
    async def _query_route_info(
        self, 
        network_info_service: NetworkInfoService, 
        source: str, 
        target: str
    ) -> Optional[RouteStatusResponse]:
        """查询两节点间的路由信息
        
        Args:
            network_info_service: 网络信息服务实例
            source: 源节点
            target: 目标节点
            
        Returns:
            Optional[RouteStatusResponse]: 路由状态响应
        """
        # 获取最新的网络报告
        report = network_info_service.get_latest_report(source)
        if not report:
            self.logger.warning(f"未找到源节点 {source} 的网络报告")
            return None
        
        # 查找目标节点信息
        target_info = None
        for dest in report.destinations:
            if dest.host_name == target:
                target_info = dest
                break
        
        if not target_info:
            self.logger.warning(f"源节点 {source} 到目标节点 {target} 不可达")
            return RouteStatusResponse(
                source_node=source,
                target_node=target,
                connection_status="unavailable",
                latency=float('inf'),  # 使用无穷大表示不可达
                bandwidth=0.0,
                route_path=[],
                response_timestamp=datetime.now()
            )
        
        # 判断连接状态
        if len(target_info.path) <= 1:
            connection_status = "direct"
        else:
            connection_status = "indirect"
        
        # 构建路由状态响应
        return RouteStatusResponse(
            source_node=source,
            target_node=target,
            connection_status=connection_status,
            latency=target_info.latency,
            bandwidth=target_info.bandwidth,
            route_path=target_info.path,
            response_timestamp=datetime.now()
        )

    async def handle_websocket_request(self, websocket, request_data: Dict[str, Any]) -> None:
        """处理WebSocket请求
        
        严格按照路由信息查询接口规范处理请求和响应
        
        Args:
            websocket: WebSocket连接
            request_data: 请求数据，必须包含request_id字段
        """
        try:
            # 验证请求数据必须包含request_id
            if 'request_id' not in request_data:
                raise ValueError("请求数据必须包含request_id字段")
                
            # 解析请求
            request = RouteInfoRequest(**request_data)
            
            # 获取路由信息
            response = await self.get_route_info(request)
            
            # 发送响应
            await websocket.send_text(response.json())
            self.logger.info(f"已发送路由信息响应: request_id={response.request_id}, 路由数量={len(response.routes)}")
            
        except Exception as e:
            self.logger.error(f"处理WebSocket请求时出错: {str(e)}")
            # 发送错误响应
            error_response = {
                "error": str(e),
                "request_id": request_data.get("request_id", "unknown")
            }
            await websocket.send_text(json.dumps(error_response))


# 创建全局服务实例
route_info_service = RouteInfoService() 