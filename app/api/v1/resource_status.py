import asyncio
import logging
from datetime import datetime
from typing import Dict, List
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse

from app.schemas.resource_status import ResourceStatusRequest, ResourceStatusInfo, ResourceStatus
from app.core.websocket_manager import WebSocketManager

# 创建路由器
router = APIRouter(prefix="/resource-status", tags=["资源状态服务"])

# 创建WebSocket连接管理器
ws_manager = WebSocketManager()

# 创建日志记录器
logger = logging.getLogger(__name__)

# 存储节点资源状态的字典
node_status: Dict[str, ResourceStatusInfo] = {}


@router.websocket("/ws")
async def resource_status_websocket(websocket: WebSocket):
    """
    资源状态WebSocket服务接口
    
    用于实时监控和更新资源状态信息的WebSocket连接
    支持双向通信，可以接收资源状态更新和发送资源状态查询请求
    
    请求格式示例:
    1. 更新资源状态:
    {
        "request_id": "req-123456",
        "node_id": "node-001",
        "status_id": "status-001",
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
        "overall_load": 58.2
    }
    
    2. 查询资源状态:
    {
        "request_id": "req-123456",
        "query": "all"  # 或特定节点ID
    }
    """
    try:
        # 接受WebSocket连接
        await ws_manager.connect(websocket)
        client_id = str(uuid4())
        logger.info(f"新的WebSocket连接已建立 - 客户端ID: {client_id}")

        while True:
            try:
                # 接收消息
                data = await websocket.receive_json()
                
                # 解析请求
                request = ResourceStatusRequest(
                    request_id=data.get("request_id", str(uuid4())),
                    request_time=datetime.now(),
                )

                # 如果收到的是资源状态更新
                if "node_id" in data:
                    try:
                        # 验证和处理资源状态信息
                        status_info = ResourceStatusInfo(
                            node_id=data["node_id"],
                            status_id=data.get("status_id", str(uuid4())),
                            timestamp=datetime.now(),
                            cpu_status=data.get("cpu_status", ResourceStatus.AVAILABLE),
                            cpu_usage=float(data.get("cpu_usage", 0)),
                            gpu_status=data.get("gpu_status", ResourceStatus.AVAILABLE),
                            gpu_usage=float(data.get("gpu_usage", 0)),
                            memory_status=data.get("memory_status", ResourceStatus.AVAILABLE),
                            memory_usage=float(data.get("memory_usage", 0)),
                            disk_status=data.get("disk_status", ResourceStatus.AVAILABLE),
                            disk_usage=float(data.get("disk_usage", 0)),
                            network_status=data.get("network_status", ResourceStatus.AVAILABLE),
                            network_usage=float(data.get("network_usage", 0)),
                            overall_load=float(data.get("overall_load", 0)),
                            response_timestamp=datetime.now()
                        )
                        
                        # 存储状态信息
                        node_status[status_info.node_id] = status_info
                        
                        # 广播更新到所有连接的客户端
                        await ws_manager.broadcast(status_info.model_dump())
                        logger.info(f"资源状态已更新 - 节点ID: {status_info.node_id}")
                        
                        # 发送确认响应
                        await websocket.send_json({
                            "request_id": request.request_id,
                            "message": "资源状态更新成功",
                            "timestamp": datetime.now().isoformat()
                        })
                        
                    except Exception as e:
                        error_msg = f"资源状态更新失败: {str(e)}"
                        logger.error(error_msg)
                        await websocket.send_json({
                            "request_id": request.request_id,
                            "error_message": error_msg,
                            "timestamp": datetime.now().isoformat()
                        })
                
                # 如果是查询请求
                elif "query" in data:
                    try:
                        if data["query"] == "all":
                            # 返回所有节点的状态
                            response = {
                                "request_id": request.request_id,
                                "nodes": [status.model_dump() for status in node_status.values()],
                                "timestamp": datetime.now().isoformat()
                            }
                        else:
                            # 返回特定节点的状态
                            node_id = data["query"]
                            if node_id in node_status:
                                response = {
                                    "request_id": request.request_id,
                                    "node": node_status[node_id].model_dump(),
                                    "timestamp": datetime.now().isoformat()
                                }
                            else:
                                response = {
                                    "request_id": request.request_id,
                                    "error_message": f"节点 {node_id} 未找到",
                                    "timestamp": datetime.now().isoformat()
                                }
                        
                        await websocket.send_json(response)
                        logger.info(f"资源状态查询已响应 - 请求ID: {request.request_id}")
                        
                    except Exception as e:
                        error_msg = f"资源状态查询失败: {str(e)}"
                        logger.error(error_msg)
                        await websocket.send_json({
                            "request_id": request.request_id,
                            "error_message": error_msg,
                            "timestamp": datetime.now().isoformat()
                        })

            except WebSocketDisconnect:
                await ws_manager.disconnect(websocket)
                logger.info(f"WebSocket连接已断开 - 客户端ID: {client_id}")
                break
                
            except Exception as e:
                logger.error(f"处理WebSocket消息时发生错误: {str(e)}")
                try:
                    await websocket.send_json({
                        "error_message": f"消息处理错误: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    })
                except:
                    break

    except Exception as e:
        logger.error(f"WebSocket连接发生错误: {str(e)}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except:
            pass


@router.get("/health")
async def health_check() -> JSONResponse:
    """
    健康检查接口
    
    返回服务的健康状态和当前活动连接数
    """
    return JSONResponse(
        content={
            "status": "healthy",
            "active_connections": len(ws_manager.active_connections),
            "timestamp": datetime.now().isoformat()
        },
        status_code=status.HTTP_200_OK
    ) 