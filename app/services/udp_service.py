"""UDP通信服务

提供与算力路由进行UDP通信的功能，支持发送算力量化请求和接收响应
"""
import asyncio
import socket
import time
from typing import List, Tuple, Optional
from loguru import logger
from app.core.config import settings



class UDPService:
    """UDP通信服务类"""

    def __init__(self):
        """初始化UDP服务"""
        # 确定性路由软件地址
        self.router_host = settings.ROUTER_HOST
        self.router_port = settings.ROUTER_PORT
        
        # 设置超时和重试参数
        self.timeout = settings.UDP_TIMEOUT
        self.max_retries = settings.UDP_RETRY_COUNT
        self.retry_delay = settings.UDP_BACKOFF
        self.exponential_backoff = settings.UDP_EXPONENTIAL_BACKOFF
        
        # 创建UDP套接字
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(self.timeout)
        except socket.error as e:
            logger.error(f"创建UDP套接字失败: {e}")
            raise

    async def send_computing_power_request(self, tasks_data: List[Tuple[int, int, int, int, int]]) -> Optional[bytes]:
        """
        发送业务算力量化请求，带有重试机制
        
        接口标识符: Task-Computing-Force-Request
        向确定性路由软件发送业务算力量化请求
        
        按照接口规范:
        数据集名称：算力业务信息
        数据元素名称: 算力资源需求量
        标识符: data-volume
        数据长度: 1+5*N字节
        
        Args:
            tasks_data: 任务算力需求数据列表，每个元素为元组(cpu_freq, cpu_cores, gpu_cores, gpu_freq, fpga_units)
            
        Returns:
            Optional[bytes]: 响应数据，如果请求失败则返回None
        """
        logger.info(f"准备发送算力量化请求，任务数量: {len(tasks_data)}")
        
        # 构建请求数据
        try:
            request_data = self._build_computing_power_request(tasks_data)
            logger.debug(f"构建的请求数据长度: {len(request_data)} 字节")
        except Exception as e:
            logger.error(f"构建算力量化请求数据失败: {str(e)}")
            return None
        
        # 重试逻辑
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries:
            try:
                # 计算当前重试的延迟时间
                if retry_count > 0:
                    delay = self.retry_delay
                    if self.exponential_backoff:
                        delay *= (2 ** (retry_count - 1))  # 指数退避
                    logger.info(f"第 {retry_count + 1} 次重试，等待 {delay:.1f} 秒")
                    await asyncio.sleep(delay)
                
                # 记录发送时间
                start_time = time.time()
                
                # 发送请求
                logger.info(f"发送算力量化请求到 {self.router_host}:{self.router_port}，尝试 {retry_count+1}/{self.max_retries}")
                self.sock.sendto(request_data, (self.router_host, self.router_port))
                
                # 等待响应
                response, addr = self.sock.recvfrom(1024)  # 假设响应数据不超过1024字节
                elapsed = (time.time() - start_time) * 1000  # 转换为毫秒
                
                logger.info(f"收到来自 {addr} 的算力量化响应，长度 {len(response)} 字节，耗时 {elapsed:.1f}ms")
                
                # 验证响应数据
                if self._validate_response(response, len(tasks_data)):
                    return response
                else:
                    logger.warning("收到无效的响应数据，将重试")
                    retry_count += 1
                    continue
                
            except socket.timeout:
                last_error = "请求超时"
                logger.warning(f"等待响应超时，尝试 {retry_count+1}/{self.max_retries}")
                retry_count += 1
            except Exception as e:
                last_error = str(e)
                logger.error(f"发送算力量化请求失败: {str(e)}")
                retry_count += 1
        
        # 所有重试都失败
        logger.error(f"达到最大重试次数 {self.max_retries}，最后一次错误: {last_error}")
        return None
        
    def _build_computing_power_request(self, tasks_data: List[Tuple[int, int, int, int, int]]) -> bytes:
        """
        构建业务算力量化请求数据
        
        按照Task-Computing-Force-Request接口规范:
        字节1表示任务数量N
        字节2~3表示该任务中所需CPU算力资源，字节2表示所需CPU主频，字节3表示所需CPU核心数
        字节4~5表示该任务所需GPU算力资源，字节4表示所需CUDA核心数，字节5表示核心频率
        字节6表示所需FPGA算力资源，具体为逻辑单元数量
        
        Args:
            tasks_data: 任务算力需求数据列表，每个元素为元组(cpu_freq, cpu_cores, gpu_cores, gpu_freq, fpga_units)
            
        Returns:
            bytes: 构建好的请求数据
        """
        # 计算任务数量
        num_tasks = len(tasks_data)
        logger.debug(f"构建请求数据，任务数量: {num_tasks}")
        
        # 构建请求数据
        request_data = bytearray([num_tasks])  # 第一个字节表示任务数量
        
        # 添加每个任务的算力需求
        for i, (cpu_freq, cpu_cores, gpu_cores, gpu_freq, fpga_units) in enumerate(tasks_data, 1):
            request_data.extend([
                cpu_freq & 0xFF,     # 字节2: CPU主频(GHz)
                cpu_cores & 0xFF,    # 字节3: CPU核心数
                gpu_cores & 0xFF,    # 字节4: GPU CUDA核心数
                gpu_freq & 0xFF,     # 字节5: GPU核心频率
                fpga_units & 0xFF    # 字节6: FPGA逻辑单元数
            ])
            
            logger.debug(
                f"任务 {i}/{num_tasks} 算力需求: "
                f"CPU主频={cpu_freq}GHz, CPU核心={cpu_cores}, "
                f"GPU核心数={gpu_cores}, GPU频率={gpu_freq}GHz, "
                f"FPGA单元={fpga_units}K"
            )
        
        return bytes(request_data)
    
    def _validate_response(self, response_data: bytes, expected_tasks: int) -> bool:
        """
        验证响应数据是否有效
        
        Args:
            response_data: 响应数据
            expected_tasks: 期望的任务数量
            
        Returns:
            bool: 响应数据是否有效
        """
        if not response_data or len(response_data) < 1:
            logger.error("响应数据为空或长度过短")
            return False
            
        # 检查任务数量
        num_tasks = response_data[0]
        if num_tasks != expected_tasks:
            logger.error(f"响应中的任务数量与请求不符: 期望 {expected_tasks}，实际 {num_tasks}")
            return False
            
        # 检查数据长度
        expected_length = 1 + 6 * num_tasks
        if len(response_data) != expected_length:
            logger.error(f"响应数据长度不匹配: 期望 {expected_length} 字节，实际 {len(response_data)} 字节")
            return False
            
        return True
    
    def parse_computing_power_response(self, response_data: bytes) -> List[Tuple[int, float, float]]:
        """
        解析业务算力度量响应数据
        
        接口标识符: Task-Computing-Force-Measurement-Result
        
        按照接口规范:
        字节1表示任务数量N
        字节2~3表示所需CPU计算能力，用核心数与主频的乘积来衡量
        字节4~5表示所需GPU计算能力，字节4为整数部分，字节5为小数部分，单位TFLOPS
        字节6~7表示所需FPGA计算能力，字节6为整数部分，字节7为小数部分，单位TFLOPS
        
        Args:
            response_data: 响应数据
            
        Returns:
            List[Tuple[int, float, float]]: 解析后的量化算力数据，每个元素为元组(cpu_power, gpu_power, fpga_power)
            cpu_power: CPU计算能力，核心数与主频的乘积
            gpu_power: GPU计算能力，单位TFLOPS
            fpga_power: FPGA计算能力，单位TFLOPS
        """
        if not response_data or len(response_data) < 1:
            logger.error("收到无效的业务算力量化响应")
            return []
        
        # 解析任务数量
        num_tasks = response_data[0]
        logger.debug(f"解析响应数据，任务数量: {num_tasks}")
        
        # 验证数据长度
        expected_length = 1 + 6 * num_tasks
        if len(response_data) != expected_length:
            logger.error(f"响应数据长度不匹配: 期望 {expected_length} 字节，实际 {len(response_data)} 字节")
            return []
        
        result = []
        offset = 1  # 跳过第一个字节(任务数量)
        
        for task_index in range(num_tasks):
            # 解析CPU计算能力
            cpu_power = (response_data[offset] << 8) | response_data[offset + 1]
            offset += 2
            
            # 解析GPU计算能力
            gpu_int = response_data[offset]
            gpu_dec = response_data[offset + 1]
            gpu_power = float(f"{gpu_int}.{gpu_dec}")
            offset += 2
            
            # 解析FPGA计算能力
            fpga_int = response_data[offset]
            fpga_dec = response_data[offset + 1]
            fpga_power = float(f"{fpga_int}.{fpga_dec}")
            offset += 2
            
            logger.debug(
                f"任务 {task_index + 1}/{num_tasks} 量化结果: "
                f"CPU={cpu_power}, "
                f"GPU={gpu_power} TFLOPS, "
                f"FPGA={fpga_power} TFLOPS"
            )
            
            result.append((cpu_power, gpu_power, fpga_power))
        
        return result 