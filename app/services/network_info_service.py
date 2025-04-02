"""网络信息服务

处理确定性算力路由软件发送的网络信息
"""
import asyncio
import struct
from typing import Dict, Optional, Tuple
from datetime import datetime
from loguru import logger

from app.core.config import settings
from app.schemas.network_info import NetworkReport, DestinationNode, NetworkInformation


class NetworkInfoService:
    """网络信息服务类"""

    def __init__(self):
        """初始化网络信息服务"""
        self.transport = None
        self.latest_reports: Dict[str, NetworkReport] = {}  # 存储最新的网络信息报告
        self.report_timestamps: Dict[str, datetime] = {}  # 存储报告时间戳
        self.logger = logger
        
        # 添加等待报告相关的成员变量
        self.report_received = asyncio.Event()
        self.waiting_for_report = False

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.start_server()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.stop_server()

    async def start_server(self, host: str = None, port: int = None):
        """启动UDP服务器监听网络信息报告
        
        Args:
            host: 监听地址，默认使用配置中的UDP_HOST
            port: 监听端口，默认使用配置中的UDP_PORT
            
        Returns:
            bool: 是否成功启动服务器
        """
        if host is None:
            host = settings.UDP_HOST
        if port is None:
            port = settings.UDP_PORT
            
        max_retries = 3
        retry_delay = 1  # 秒
        
        for retry in range(max_retries):
            try:
                loop = asyncio.get_running_loop()
                # 尝试使用备用端口如果主端口被占用
                actual_port = port + retry if retry > 0 else port
                
                self.transport, _ = await loop.create_datagram_endpoint(
                    lambda: self,
                    local_addr=(host, actual_port)
                )
                self.logger.info(f"UDP服务器已启动，监听 {host}:{actual_port}")
                
                # 初始化服务状态
                self.report_received.clear()
                self.waiting_for_report = False
                self.latest_reports = {}
                self.report_timestamps = {}
                
                return True
            except OSError as e:
                if "地址已经被使用" in str(e) or "Address already in use" in str(e):
                    if retry < max_retries - 1:
                        self.logger.warning(f"端口 {actual_port} 已被占用，尝试使用端口 {port + retry + 1}")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
                        continue
                    else:
                        self.logger.error(f"所有备用端口都被占用，无法启动UDP服务器")
                        return False
                else:
                    self.logger.error(f"启动UDP服务器失败: {str(e)}")
                    return False
            except Exception as e:
                self.logger.error(f"启动UDP服务器失败: {str(e)}")
                
                # 清理资源
                if self.transport:
                    try:
                        self.transport.close()
                    except:
                        pass
                    self.transport = None
                    
                return False

    def connection_made(self, transport):
        """建立连接时的回调"""
        self.transport = transport
        self.logger.debug("UDP服务器连接已建立")

    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        """接收UDP数据包的回调
        
        接口标识符: Network-Information-Report
        解析确定性算力路由软件发送的网络信息数据包
        
        Args:
            data: 收到的数据
            addr: 发送方地址和端口
        """
        self.logger.debug(f"收到来自 {addr} 的数据包，长度: {len(data)}字节")
        
        try:
            # 解析报告ID (4字节)
            report_id, = struct.unpack("!I", data[0:4])
            
            # 解析镜像仓库节点ID (64字节)
            mirror_repo_host_bytes = data[4:68]
            mirror_repo_host = mirror_repo_host_bytes.split(b'\x00')[0].decode('utf-8')
            
            # 解析目的节点数量 (1字节)
            dest_host_number = data[68]
            
            self.logger.debug(f"报告ID: {report_id}, 镜像仓库: {mirror_repo_host}, 目标节点数量: {dest_host_number}")
            
            # 解析目标节点信息
            destinations = []
            offset = 69  # 从第69个字节开始解析目标节点信息
            
            for i in range(dest_host_number):
                # 解析目的节点ID (64字节)
                dest_host_name_bytes = data[offset:offset+64]
                # 去掉填充的空字节
                dest_host_name = dest_host_name_bytes.split(b'\x00')[0].decode('utf-8')
                offset += 64
                
                # 解析传输时延 (2字节)
                latency, = struct.unpack("!H", data[offset:offset+2])
                offset += 2
                
                # 解析可用带宽 (3字节)
                bandwidth_int, = struct.unpack("!H", data[offset:offset+2])
                bandwidth_dec = data[offset+2]
                bandwidth = float(f"{bandwidth_int}.{bandwidth_dec}")
                offset += 3
                
                # 创建网络信息对象
                network_info = NetworkInformation(
                    latency=latency,
                    bandwidth=bandwidth
                )
                
                # 创建目标节点对象
                dest_node = DestinationNode(
                    dest_host_name=dest_host_name,
                    network_info=network_info
                )
                
                destinations.append(dest_node)
                
                self.logger.debug(f"节点 {dest_host_name}: 延迟={latency}ms, 带宽={bandwidth}Mbps")
            
            # 创建网络报告对象
            report = NetworkReport(
                report_id=report_id,
                mirror_repository_host_name=mirror_repo_host,
                dest_host_number=dest_host_number,
                destinations=destinations,
                timestamp=datetime.now()
            )
            
            # 存储报告和时间戳
            self.latest_reports[mirror_repo_host] = report
            self.report_timestamps[mirror_repo_host] = datetime.now()
            
            self.logger.info(f"成功解析来自 {mirror_repo_host} 的网络信息报告，包含 {dest_host_number} 个目标节点")
            
            # 如果有等待报告的任务，通知它们
            if self.waiting_for_report:
                self.report_received.set()
                
        except Exception as e:
            self.logger.error(f"解析网络信息报告时出错: {str(e)}")
            self.logger.exception(e)

    def error_received(self, exc):
        """接收错误的回调"""
        self.logger.error(f"UDP服务器接收错误: {str(exc)}")

    def connection_lost(self, exc):
        """连接断开时的回调"""
        self.logger.warning(f"UDP服务器连接断开: {str(exc) if exc else '正常断开'}")

    def get_latest_report(self, mirror_repo_host: str) -> Optional[NetworkReport]:
        """
        获取指定镜像仓库节点的最新网络信息报告
        
        Args:
            mirror_repo_host: 镜像仓库节点ID
            
        Returns:
            Optional[NetworkReport]: 网络信息报告，如果不存在则返回None
        """
        return self.latest_reports.get(mirror_repo_host)

    def get_report_timestamp(self, mirror_repo_host: str) -> Optional[datetime]:
        """
        获取指定镜像仓库节点的最新报告时间
        
        Args:
            mirror_repo_host: 镜像仓库节点ID
            
        Returns:
            Optional[datetime]: 报告时间，如果不存在则返回None
        """
        return self.report_timestamps.get(mirror_repo_host)
    
    async def wait_for_report(self, timeout: float = None) -> bool:
        """
        等待接收网络信息上报
        
        Args:
            timeout: 超时时间（秒），默认使用配置中的UDP_TIMEOUT
            
        Returns:
            bool: 是否成功接收到报告
        """
        try:
            if timeout is None:
                timeout = settings.UDP_TIMEOUT
            
            # 如果服务未初始化或事件未初始化，直接返回False
            if not hasattr(self, 'report_received') or self.report_received is None:
                self.logger.warning("网络信息服务未正确初始化，无法等待报告")
                return False
                
            self.waiting_for_report = True
            self.report_received.clear()
            await asyncio.wait_for(self.report_received.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            self.logger.warning(f"等待网络信息上报超时: {timeout}秒")
            return False
        except Exception as e:
            self.logger.error(f"等待网络信息上报时发生错误: {str(e)}")
            return False
        finally:
            self.waiting_for_report = False

    async def stop_server(self):
        """停止UDP服务器"""
        if self.transport:
            self.transport.close()
            self.logger.info("UDP服务器已停止") 