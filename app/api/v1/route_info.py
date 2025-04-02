"""路由信息查询API路由

实现基于WebSocket的路由信息查询接口（Route-Info-Query）
"""
import json
from typing import Dict, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, status
from loguru import logger

from app.schemas.route_info import RouteInfoRequest, RouteInfoResponse
from app.services.route_info_service import route_info_service
from app.core.config import settings

# 创建路由
router = APIRouter(prefix="/route-info", tags=["route_info"])


class ConnectionManager:
    """WebSocket连接管理器
    
    """
    
    def __init__(self):
        """初始化连接管理器"""
        self.active_connection: Optional[WebSocket] = None
        
    async def connect(self, websocket: WebSocket):
        """建立连接
        
        Args:
            websocket: WebSocket连接
        """
        await websocket.accept()
        self.active_connection = websocket
        logger.info("算力路由已连接")
    
    def disconnect(self):
        """断开连接"""
        if self.active_connection:
            self.active_connection = None
            logger.info("算力路由已断开连接")
    
    async def send_message(self, message: str):
        """向客户端发送消息
        
        Args:
            message: 消息内容
        """
        if self.active_connection:
            await self.active_connection.send_text(message)


# 创建连接管理器实例
manager = ConnectionManager()


@router.websocket("/ws")
async def route_info_websocket(websocket: WebSocket):
    """路由信息查询WebSocket接口
    
    接口标识符: Route-Info-Query
    
    处理基于WebSocket的路由信息查询请求，只与算力路由层进行交互
    
    Args:
        websocket: WebSocket连接
    """
    await manager.connect(websocket)
    try:
        # 发送欢迎消息
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "message": "已连接到路由信息查询服务"
        }))
        
        # 循环接收消息
        while True:
            # 接收JSON格式的请求数据
            data = await websocket.receive_text()
            logger.info(f"收到算力路由请求: {data}")
            
            try:
                # 解析请求数据
                request_data = json.loads(data)
                
                # 处理请求
                await route_info_service.handle_websocket_request(websocket, request_data)
                
            except json.JSONDecodeError:
                logger.error(f"无法解析JSON数据: {data}")
                await websocket.send_text(json.dumps({
                    "error": "无效的JSON格式",
                }))
            except Exception as e:
                logger.error(f"处理WebSocket消息时出错: {str(e)}")
                await websocket.send_text(json.dumps({
                    "error": f"处理请求失败: {str(e)}",
                }))
    
    except WebSocketDisconnect:
        logger.info("算力路由断开连接")
    finally:
        manager.disconnect()


@router.get("/api-info", response_model=Dict[str, Any])
async def get_route_info_api_info():
    """获取路由信息查询API信息
    
    返回路由信息查询API的相关信息，包括接口标识符、协议类型等
    
    Returns:
        Dict[str, Any]: API信息
    """
    return {
        "api_name": "路由信息查询接口",
        "api_identifier": "Route-Info-Query",
        "protocol": "WebSocket over HTTPS(WSS)",
        "websocket_endpoint": f"wss://{settings.HOST}:{settings.PORT}{settings.API_V1_STR}/route-info/ws",
        "description": "提供节点间路由信息的实时查询功能，仅与算力路由层交互",
        "version": "1.0"
    } 