from typing import List, Dict, Any
import logging
from fastapi import WebSocket, status

logger = logging.getLogger(__name__)

class WebSocketManager:
    """
    WebSocket连接管理器
    
    用于管理所有活动的WebSocket连接，支持:
    - 连接的建立和断开
    - 广播消息到所有连接
    - 发送消息到特定连接
    - 自动清理断开的连接
    """
    
    def __init__(self):
        """初始化WebSocket连接管理器"""
        self.active_connections: List[WebSocket] = []
        self._connection_count = 0
    
    async def connect(self, websocket: WebSocket) -> None:
        """
        建立新的WebSocket连接
        
        Args:
            websocket: WebSocket连接实例
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        self._connection_count += 1
        logger.info(f"新WebSocket连接已建立 - 当前活动连接数: {self._connection_count}")
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """
        断开WebSocket连接
        
        Args:
            websocket: 要断开的WebSocket连接实例
        """
        try:
            self.active_connections.remove(websocket)
            self._connection_count -= 1
            logger.info(f"WebSocket连接已断开 - 当前活动连接数: {self._connection_count}")
        except ValueError:
            logger.warning("尝试断开不存在的WebSocket连接")
    
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """
        广播消息到所有活动连接
        
        Args:
            message: 要广播的消息内容
        """
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"广播消息失败: {str(e)}")
                disconnected.append(connection)
        
        # 清理断开的连接
        for connection in disconnected:
            await self.disconnect(connection)
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket) -> None:
        """
        发送消息到特定连接
        
        Args:
            message: 要发送的消息内容
            websocket: 目标WebSocket连接实例
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"发送个人消息失败: {str(e)}")
            await self.disconnect(websocket)
    
    @property
    def active_connections_count(self) -> int:
        """获取当前活动连接数"""
        return self._connection_count 