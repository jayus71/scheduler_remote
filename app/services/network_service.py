"""网络信息服务模块"""
import socket
import struct
import asyncio
from typing import Optional, Tuple, List
from loguru import logger

from app.schemas.network import NetworkReport, NetworkInfo, DestinationHost, NetworkInfoStore


class NetworkService:
    """网络信息服务类"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 9000):
        """初始化网络信息服务
        
        Args:
            host: 监听地址，默认为所有地址
            port: 监听端口，默认为9000
        """
        self.host = host
        self.port = port
        self.network_info_store = NetworkInfoStore()
        self.transport = None
        self.protocol = None
        self.is_running = False
        
    async def start(self):
        """启动UDP服务器"""
        if self.is_running:
            logger.warning("网络信息服务已经在运行中")
            return
            
        loop = asyncio.get_event_loop()
        
        try:
            self.transport, self.protocol = await loop.create_datagram_endpoint(
                lambda: UDPServerProtocol(self._process_datagram),
                local_addr=(self.host, self.port)
            )
            self.is_running = True
            logger.info(f"网络信息服务已启动，监听地址: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"启动网络信息服务失败: {str(e)}")
            raise
    
    async def start_server(self):
        """启动UDP服务器（别名方法）"""
        await self.start()
    
    async def stop(self):
        """停止UDP服务器"""
        if not self.is_running:
            logger.warning("网络信息服务未运行")
            return
            
        if self.transport:
            self.transport.close()
            self.transport = None
            self.protocol = None
            self.is_running = False
            logger.info("网络信息服务已停止")
    
    async def stop_server(self):
        """停止UDP服务器（别名方法）"""
        await self.stop()
    
    def _process_datagram(self, data: bytes, addr: Tuple[str, int]):
        """处理接收到的UDP数据包
        
        Args:
            data: 接收到的数据
            addr: 发送方地址
        """
        try:
            report = self._parse_network_report(data)
            if report:
                self.network_info_store.update_network_info(report)
                logger.info(f"收到来自 {addr} 的网络信息报告，报告ID: {report.report_id}，目的节点数量: {len(report.destinations)}")
        except Exception as e:
            logger.error(f"处理网络信息报告失败: {str(e)}")
    
    def _parse_network_report(self, data: bytes) -> Optional[NetworkReport]:
        """解析网络信息报告数据
        
        根据协议格式解析UDP数据包
        
        Args:
            data: UDP数据包内容
            
        Returns:
            NetworkReport: 解析后的网络信息报告，解析失败则返回None
        """
        if len(data) < 69:  # 最小长度: 4(报告ID) + 64(仓库节点ID) + 1(目的节点数量)
            logger.error(f"数据包长度不足，无法解析: {len(data)} bytes")
            return None
        
        try:
            # 解析报告ID (4字节)
            report_id = struct.unpack('!I', data[0:4])[0]
            
            # 解析镜像仓库节点ID (64字节字符串)
            repository_host_raw = data[4:68]
            repository_host = repository_host_raw.decode('utf-8').rstrip('\x00')
            
            # 解析目的节点数量 (1字节)
            dest_host_number = data[68]
            
            # 初始化网络报告
            report = NetworkReport(
                report_id=report_id,
                mirror_repository_host_name=repository_host,
                destinations=[]
            )
            
            # 解析目的节点信息
            offset = 69
            for i in range(dest_host_number):
                if offset + 74 > len(data):  # 检查剩余数据是否足够
                    logger.error(f"数据包长度不足，无法解析目的节点 {i+1}/{dest_host_number}")
                    break
                
                # 解析目的节点ID (64字节字符串)
                dest_host_raw = data[offset:offset+64]
                dest_host = dest_host_raw.decode('utf-8').rstrip('\x00')
                offset += 64
                
                # 解析网络信息 (至少需要5字节: 2字节传输时延 + 2字节带宽整数 + 1字节带宽小数)
                if offset + 5 > len(data):
                    logger.error(f"数据包长度不足，无法解析节点 {dest_host} 的网络信息")
                    break
                
                # 解析传输时延 (2字节)
                latency = struct.unpack('!H', data[offset:offset+2])[0]
                offset += 2
                
                # 解析带宽 (整数部分2字节，小数部分1字节)
                bandwidth_integer = struct.unpack('!H', data[offset:offset+2])[0]
                offset += 2
                bandwidth_decimal = data[offset]
                offset += 1
                
                # 创建目的节点和网络信息对象
                network_info = NetworkInfo(
                    latency=latency,
                    bandwidth_integer=bandwidth_integer,
                    bandwidth_decimal=bandwidth_decimal
                )
                
                destination = DestinationHost(
                    host_name=dest_host,
                    network_info=network_info
                )
                
                report.destinations.append(destination)
            
            return report
            
        except Exception as e:
            logger.error(f"解析网络信息报告失败: {str(e)}", exc_info=True)
            return None
    
    def get_network_info(self, source: str, dest: str) -> Optional[NetworkInfo]:
        """获取从源节点到目标节点的网络信息
        
        Args:
            source: 源节点
            dest: 目标节点
            
        Returns:
            NetworkInfo: 网络信息，如果不存在则返回None
        """
        return self.network_info_store.get_network_info(source, dest)
    
    def get_all_network_info(self):
        """获取所有网络信息
        
        Returns:
            Dict: 所有网络信息的字典
        """
        return self.network_info_store.reports


class UDPServerProtocol(asyncio.DatagramProtocol):
    """UDP服务器协议类"""
    
    def __init__(self, message_callback):
        """初始化UDP服务器协议
        
        Args:
            message_callback: 接收消息的回调函数
        """
        self.message_callback = message_callback
        
    def connection_made(self, transport):
        """连接建立时的回调
        
        Args:
            transport: 传输对象
        """
        pass
        
    def datagram_received(self, data, addr):
        """接收到数据包时的回调
        
        Args:
            data: 接收到的数据
            addr: 发送方地址
        """
        self.message_callback(data, addr)
        
    def error_received(self, exc):
        """发生错误时的回调
        
        Args:
            exc: 异常对象
        """
        logger.error(f"UDP服务器错误: {str(exc)}")
        
    def connection_lost(self, exc):
        """连接断开时的回调
        
        Args:
            exc: 异常对象，如果是正常关闭则为None
        """
        if exc:
            logger.error(f"UDP服务器连接断开: {str(exc)}")
        else:
            logger.info("UDP服务器连接正常关闭") 