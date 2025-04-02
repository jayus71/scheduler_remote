"""算力服务

处理业务算力需求解析和量化结果处理
"""
from typing import List, Dict, Tuple, Optional
from loguru import logger
import asyncio
import struct
from datetime import datetime
import logging

from app.schemas.common import Container
from app.services.udp_service import UDPService
from app.core.config import settings
from app.utils.resource_parser import parse_resource_value
from app.schemas.computing_power import (
    ComputingForceType,
    CPUInformation,
    GPUInformation,
    FPGAInformation,
    NodeComputingForce,
    ComputingForceReport
)


class ComputingPowerService:
    """算力服务类，用于分析和量化计算需求"""

    def __init__(self):
        """初始化算力服务"""
        self.udp_service = UDPService()
        self.enable_computing_power = settings.ENABLE_COMPUTING_POWER
        self.fallback_strategy = settings.COMPUTING_POWER_FALLBACK
        self.default_cpu_power_ratio = settings.DEFAULT_CPU_POWER_RATIO
        self.default_gpu_tflops = settings.DEFAULT_GPU_TFLOPS
        self.default_fpga_tflops = settings.DEFAULT_FPGA_TFLOPS
        self.logger = logger
        self.transport = None
        self.latest_reports: Dict[str, NodeComputingForce] = {}  # 存储最新的节点算力信息
        self.report_timestamps: Dict[str, datetime] = {}  # 存储报告时间戳
        self.waiting_for_report = False  # 是否正在等待算力信息上报
        self.report_received = asyncio.Event()  # 用于通知算力信息已收到
        
    async def analyze_computing_power_requirements(
        self,
        containers: List
    ) -> List[Tuple[int, int, int, int, int]]:
        """分析容器的算力需求
        
        Args:
            containers: 容器列表，每个容器可以是Container对象或字典
        
        Returns:
            List[Tuple[int, int, int, int, int]]: [(CPU主频, CPU核心数, GPU核心数, GPU频率, FPGA单元数), ...]
        """
        requirements = []
        
        for container in containers:
            try:
                # 获取资源请求
                if isinstance(container, dict):
                    resources = container.get("resources", {}).get("requests", {})
                    cpu_request = resources.get("cpu", "1")
                    gpu_request = resources.get("nvidia.com/gpu", "0")
                    fpga_request = resources.get("fpga.intel.com/arria10", "0")
                else:
                    # 假设是Container对象
                    resources = getattr(container.resources, "requests", None)
                    if resources is None:
                        cpu_request = "1"
                        gpu_request = "0"
                        fpga_request = "0"
                    else:
                        cpu_request = getattr(resources, "cpu", "1")
                        gpu_request = getattr(resources, "nvidia.com/gpu", "0")
                        fpga_request = getattr(resources, "fpga.intel.com/arria10", "0")
                
                # 分析CPU需求
                try:
                    cpu_cores = int(float(parse_resource_value(str(cpu_request))) / 1000)  # 转换为核心数
                    cpu_freq = 2  # 假设默认需要2GHz主频
                except (ValueError, TypeError):
                    self.logger.warning(f"无效的CPU请求: {cpu_request}，使用默认值")
                    cpu_cores = 1
                    cpu_freq = 1
                
                # 分析GPU需求
                try:
                    gpu_count = int(float(parse_resource_value(str(gpu_request))))
                    if gpu_count > 0:
                        gpu_cores = 1024  # 假设每个GPU有1024个CUDA核心
                        gpu_freq = 1      # 假设需要1GHz GPU频率
                    else:
                        gpu_cores = 0
                        gpu_freq = 0
                except (ValueError, TypeError):
                    self.logger.warning(f"无效的GPU请求: {gpu_request}，使用默认值")
                    gpu_cores = 0
                    gpu_freq = 0
                
                # 分析FPGA需求
                try:
                    fpga_count = int(float(parse_resource_value(str(fpga_request))))
                    if fpga_count > 0:
                        fpga_units = 100  # 假设每个FPGA有100K逻辑单元
                    else:
                        fpga_units = 0
                except (ValueError, TypeError):
                    self.logger.warning(f"无效的FPGA请求: {fpga_request}，使用默认值")
                    fpga_units = 0
                
                # 添加到需求列表
                requirements.append((cpu_freq, cpu_cores, gpu_cores, gpu_freq, fpga_units))
                
                # 记录日志
                self.logger.debug(
                    f"容器算力需求分析结果: "
                    f"CPU({cpu_freq}GHz x {cpu_cores}核), "
                    f"GPU({gpu_cores}核心 x {gpu_freq}GHz), "
                    f"FPGA({fpga_units}K逻辑单元)"
                )
            
            except Exception as e:
                self.logger.error(f"分析容器算力需求时发生错误: {str(e)}")
                # 使用默认值
                requirements.append((1, 1, 0, 0, 0))
        
        return requirements

    async def quantify_computing_power(self, containers: List) -> Optional[List[Tuple[int, float, float]]]:
        """
        量化容器的算力需求
        
        按照Task-Computing-Force-Request接口向确定性路由软件发送请求，
        等待Task-Computing-Force-Measurement-Result接口返回业务算力度量结果
        
        Args:
            containers: 容器列表，每个容器可以是Container对象或字典
            
        Returns:
            Optional[List[Tuple[int, float, float]]]: 每个容器的算力需求元组列表 (cpu_power, gpu_power, fpga_power)
            如果无法量化则返回None
        """
        try:
            # 分析算力需求
            requirements = await self.analyze_computing_power_requirements(containers)
            if not requirements:
                logger.warning("未能分析出算力需求，使用默认值")
                return self._get_default_computing_power(len(containers))
            
            logger.info(f"分析出 {len(requirements)} 个容器的算力需求，准备发送给确定性路由软件")
            
            # 发送业务算力量化请求给确定性路由软件
            # 接口标识符：Task-Computing-Force-Request
            response_data = await self.udp_service.send_computing_power_request(requirements)
            
            if not response_data:
                logger.warning("未能从确定性路由软件获取业务算力量化结果，使用降级策略")
                return self._apply_fallback_strategy(requirements)
            
            # 解析业务算力度量结果
            # 接口标识符：Task-Computing-Force-Measurement-Result
            quantified_power = self.udp_service.parse_computing_power_response(response_data)
            
            if not quantified_power:
                logger.warning("解析业务算力度量结果失败，使用降级策略")
                return self._apply_fallback_strategy(requirements)
            
            logger.info(f"成功获取业务算力量化结果: {quantified_power}")
            return quantified_power
            
        except Exception as e:
            logger.error(f"量化算力需求时发生错误: {str(e)}")
            if len(containers) > 0:
                return self._get_default_computing_power(len(containers))
            return None

    def _get_default_computing_power(
        self,
        container_count: int
    ) -> List[Tuple[int, float, float]]:
        """获取默认算力值
        
        Args:
            container_count: 容器数量
        
        Returns:
            List[Tuple[int, float, float]]: [(CPU能力, GPU能力, FPGA能力), ...]
        """
        return [(2, self.default_gpu_tflops, self.default_fpga_tflops)] * container_count

    def _apply_fallback_strategy(
        self,
        requirements: List[Tuple[int, int, int, int, int]]
    ) -> List[Tuple[int, float, float]]:
        """应用降级策略
        
        Args:
            requirements: 算力需求列表
        
        Returns:
            List[Tuple[int, float, float]]: [(CPU能力, GPU能力, FPGA能力), ...]
        """
        result = []
        
        for cpu_freq, cpu_cores, gpu_cores, gpu_freq, fpga_units in requirements:
            # 计算CPU能力（主频 x 核心数 x 效率系数）
            cpu_power = int(cpu_freq * cpu_cores * self.default_cpu_power_ratio)
            
            # 计算GPU能力（CUDA核心数 x 频率 / 1000，得到TFLOPS）
            if gpu_cores > 0 and gpu_freq > 0:
                gpu_power = float(gpu_cores * gpu_freq) / 1000
            else:
                gpu_power = 0.0
            
            # 计算FPGA能力（逻辑单元数 / 100，得到TFLOPS）
            if fpga_units > 0:
                fpga_power = float(fpga_units) / 100
            else:
                fpga_power = 0.0
            
            # 根据降级策略调整算力值
            if self.fallback_strategy == "conservative":
                # 保守策略：使用1.5倍的计算值
                cpu_power = int(cpu_power * 1.5)
                gpu_power = gpu_power * 1.5
                fpga_power = fpga_power * 1.5
            
            elif self.fallback_strategy == "aggressive":
                # 激进策略：使用2倍的计算值
                cpu_power = int(cpu_power * 2)
                gpu_power = gpu_power * 2
                fpga_power = fpga_power * 2
            
            # 默认策略：使用原始计算值
            result.append((cpu_power, gpu_power, fpga_power))
        
        return result

    async def start(self):
        """启动算力服务
        
        启动UDP服务器并初始化服务状态
        """
        try:
            # 初始化服务状态
            self.latest_reports = {}
            self.report_timestamps = {}
            self.report_received.clear()
            
            # 启动UDP服务器
            await self.start_server()
            logger.info("算力服务已启动")
            return True
        except Exception as e:
            logger.error(f"启动算力服务失败: {str(e)}")
            # 尝试关闭可能部分初始化的资源
            try:
                if self.transport:
                    self.transport.close()
                    self.transport = None
            except Exception as cleanup_err:
                logger.error(f"清理算力服务资源时出错: {str(cleanup_err)}")
            return False

    async def start_server(self):
        """启动UDP服务器"""
        max_retries = 3
        retry_delay = 1  # 秒
        
        for retry in range(max_retries):
            try:
                loop = asyncio.get_event_loop()
                # 尝试使用备用端口如果主端口被占用
                port = settings.UDP_SERVICE_PORT + retry if retry > 0 else settings.UDP_SERVICE_PORT
                
                self.transport, _ = await loop.create_datagram_endpoint(
                    lambda: self,
                    local_addr=(settings.UDP_SERVICE_HOST, port)
                )
                logger.info(f"UDP服务器已启动 - {settings.UDP_SERVICE_HOST}:{port}")
                
                # 成功启动，返回
                return
            except OSError as e:
                if "地址已经被使用" in str(e) or "Address already in use" in str(e):
                    if retry < max_retries - 1:
                        logger.warning(f"端口 {port} 已被占用，尝试使用端口 {settings.UDP_SERVICE_PORT + retry + 1}")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
                        continue
                    else:
                        logger.error(f"所有备用端口都被占用，无法启动UDP服务器")
                        raise
                else:
                    logger.error(f"启动UDP服务器失败: {str(e)}")
                    raise
            except Exception as e:
                logger.error(f"启动UDP服务器失败: {str(e)}")
                raise
            
    def connection_made(self, transport):
        """建立连接时的回调"""
        self.transport = transport
        
    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        """
        接收UDP数据报文
        
        接口标识符: Node-Computing-Force-Information-Report
        解析确定性算力路由软件发送的节点算力度量信息
        
        Args:
            data: 接收到的数据
            addr: 发送方地址
        """
        try:
            # 解析报告ID (4字节)
            report_id = struct.unpack('!I', data[0:4])[0]
            offset = 4
            
            # 解析节点数量 (1字节)
            host_number = data[offset]
            offset += 1
            
            nodes = []
            # 解析每个节点的信息
            for _ in range(host_number):
                # 解析节点ID (64字节)
                host_name = data[offset:offset+64].decode('utf-8').strip('\x00')
                offset += 64
                
                # 解析内存信息 (2字节)
                total_memory, available_memory = struct.unpack('!BB', data[offset:offset+2])
                offset += 2
                
                # 解析算力资源类型 (1字节)
                computing_force_type = data[offset]
                offset += 1
                
                node_info = {
                    'report_id': report_id,
                    'host_name': host_name,
                    'total_memory': total_memory,
                    'available_memory': available_memory,
                    'computing_force_type': computing_force_type
                }
                
                # 根据算力资源类型解析相应信息
                if computing_force_type == ComputingForceType.CPU:
                    # 解析CPU数量 (1字节)
                    cpu_count = data[offset]
                    offset += 1
                    
                    cpu_info = []
                    # 计算CPU负载平均值，用于后续打分
                    total_load = 0
                    
                    for _ in range(cpu_count):
                        # 解析CPU计算能力 (3字节)
                        computing_power = struct.unpack('!f', data[offset:offset+3] + b'\x00')[0]
                        offset += 3
                        
                        # 解析CPU负载 (1字节)
                        load = data[offset]
                        total_load += load
                        offset += 1
                        
                        cpu_info.append(CPUInformation(
                            core_count=1,  # 假设每个CPU为单核
                            computing_power=computing_power,
                            load=load
                        ))
                    
                    node_info['cpu_information'] = cpu_info
                    # 为打分计算添加平均负载
                    node_info['cpu_load_percentage'] = total_load / cpu_count if cpu_count > 0 else 0
                    
                elif computing_force_type == ComputingForceType.GPU:
                    # 解析GPU数量 (1字节)
                    gpu_count = data[offset]
                    offset += 1
                    
                    gpu_info = []
                    # 计算GPU负载平均值，用于后续打分
                    total_load = 0
                    
                    for _ in range(gpu_count):
                        # 解析GPU算力 (2字节)
                        tflops_int = data[offset]
                        tflops_dec = data[offset+1]
                        tflops = float(f"{tflops_int}.{tflops_dec}")
                        offset += 2
                        
                        # 解析显存 (2字节)
                        memory_int = data[offset]
                        memory_dec = data[offset+1]
                        memory = float(f"{memory_int}.{memory_dec}")
                        offset += 2
                        
                        # 解析GPU负载 (1字节)
                        load = data[offset]
                        total_load += load
                        offset += 1
                        
                        gpu_info.append(GPUInformation(
                            tflops=tflops,
                            memory=memory,
                            load=load
                        ))
                    
                    node_info['gpu_information'] = gpu_info
                    # 为打分计算添加平均负载
                    node_info['gpu_load_percentage'] = total_load / gpu_count if gpu_count > 0 else 0
                    
                elif computing_force_type == ComputingForceType.FPGA:
                    # 解析FPGA数量 (1字节)
                    fpga_count = data[offset]
                    offset += 1
                    
                    fpga_info = []
                    # 计算FPGA负载平均值，用于后续打分
                    total_load = 0
                    
                    for _ in range(fpga_count):
                        # 解析FPGA算力 (2字节)
                        tflops_int = data[offset]
                        tflops_dec = data[offset+1]
                        tflops = float(f"{tflops_int}.{tflops_dec}")
                        offset += 2
                        
                        # 解析FPGA负载 (1字节)
                        load = data[offset]
                        total_load += load
                        offset += 1
                        
                        fpga_info.append(FPGAInformation(
                            tflops=tflops,
                            load=load
                        ))
                    
                    node_info['fpga_information'] = fpga_info
                    # 为打分计算添加平均负载
                    node_info['fpga_load_percentage'] = total_load / fpga_count if fpga_count > 0 else 0
                
                # 创建节点算力信息对象
                node = NodeComputingForce(**node_info)
                nodes.append(node)
                
                # 更新最新报告
                self.latest_reports[host_name] = node
                self.report_timestamps[host_name] = datetime.now()
            
            # 创建完整的报告对象
            report = ComputingForceReport(
                report_id=report_id,
                host_number=host_number,
                nodes=nodes
            )
            
            logger.info(f"收到算力信息上报 - 报告ID: {report_id}, 节点数量: {host_number}")
            
            # 如果正在等待报告，设置事件通知
            if self.waiting_for_report:
                self.report_received.set()
                
        except Exception as e:
            logger.error(f"解析算力信息报文失败: {str(e)}")
            logger.exception(e)
            
    def error_received(self, exc):
        """接收错误时的回调"""
        logger.error(f"UDP服务器发生错误: {str(exc)}")
        
    def connection_lost(self, exc):
        """连接断开时的回调"""
        logger.warning(f"UDP服务器连接断开: {str(exc) if exc else '正常断开'}")
        
    def get_latest_report(self, node_name: str) -> Optional[NodeComputingForce]:
        """
        获取指定节点的最新算力信息
        
        Args:
            node_name: 节点名称
            
        Returns:
            Optional[NodeComputingForce]: 节点算力信息，如果不存在则返回None
        """
        return self.latest_reports.get(node_name)
        
    def get_report_timestamp(self, node_name: str) -> Optional[datetime]:
        """
        获取指定节点的最新报告时间
        
        Args:
            node_name: 节点名称
            
        Returns:
            Optional[datetime]: 报告时间，如果不存在则返回None
        """
        return self.report_timestamps.get(node_name)
        
    async def wait_for_report(self, timeout: float = None) -> bool:
        """
        等待接收算力信息上报
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            bool: 是否成功接收到报告
        """
        try:
            # 如果服务未初始化或事件未初始化，直接返回False
            if not hasattr(self, 'report_received') or self.report_received is None:
                logger.warning("算力服务未正确初始化，无法等待报告")
                return False
                
            self.waiting_for_report = True
            self.report_received.clear()
            await asyncio.wait_for(self.report_received.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(f"等待算力信息上报超时: {timeout}秒")
            return False
        except Exception as e:
            logger.error(f"等待算力信息上报时发生错误: {str(e)}")
            return False
        finally:
            self.waiting_for_report = False
            
    def stop_server(self):
        """停止UDP服务器"""
        if self.transport:
            self.transport.close()
            logger.info("UDP服务器已停止")

    async def stop(self):
        """停止算力服务"""
        self.stop_server()
        logger.info("算力服务已停止") 