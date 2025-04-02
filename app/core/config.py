"""配置模块"""
import os
from typing import Optional, Dict, Any
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置类"""
    
    # 应用信息
    APP_NAME: str = "星载微服务协同系统"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_V1_PREFIX: str = "/v1"
    API_V1_STR: str = "/api/v1"
    
    # Kubernetes配置
    KUBERNETES_SERVICE_HOST: Optional[str] = None
    KUBERNETES_SERVICE_PORT: Optional[str] = None
    KUBE_NAMESPACE: str = "default"
    USE_SERVICE_ACCOUNT: bool = False

    
    # 网络服务配置
    NETWORK_SERVICE_HOST: str = "0.0.0.0"
    NETWORK_SERVICE_PORT: int = 9000  # NetworkService使用的端口
    
    # 网络信息服务配置
    UDP_HOST: str = "0.0.0.0"
    UDP_PORT: int = 9001  # NetworkInfoService使用的端口
    UDP_BUFFER_SIZE: int = 1024  # UDP缓冲区大小
    UDP_TIMEOUT: int = 10  # UDP请求超时时间（秒）
    UDP_RETRY_COUNT: int = 3  # UDP请求最大重试次数
    UDP_BACKOFF: float = 1.0  # 重试间隔（秒）
    
    # Kubernetes资源限制
    DEFAULT_CPU_REQUEST: str = "100m"
    DEFAULT_MEMORY_REQUEST: str = "128Mi"
    DEFAULT_CPU_LIMIT: str = "200m"
    DEFAULT_MEMORY_LIMIT: str = "256Mi"
    
    # 调度策略配置
    SCHEDULER_FILTER_TIMEOUT: int = 5
    MAX_PODS_PER_NODE: int = 110
    ENABLE_GPU_SCHEDULING: bool = False
    
    # 优化器配置
    ENABLE_OPTIMIZER: bool = True
    WEIGHT_CPU: float = 0.6
    WEIGHT_MEMORY: float = 0.4
    OPTIMIZER_ALGORITHM: str = "default"
    
    # 遗传算法配置
    GA_POPULATION_SIZE: int = 50
    GA_GENERATIONS: int = 100
    GA_MUTATION_RATE: float = 0.1
    GA_WEIGHT_CPU: float = 0.5
    GA_WEIGHT_MEMORY: float = 0.3
    GA_WEIGHT_BALANCE: float = 0.2
    
    # 粒子群优化(PSO)算法配置
    OPTIMIZATION_METHOD: str   # 优化方法选择: genetic, pso, sa, aco
    PSO_SWARM_SIZE: int = 30  # 粒子群大小
    PSO_MAX_ITERATIONS: int = 50  # 最大迭代次数
    PSO_INERTIA_WEIGHT: float = 0.7  # 惯性权重
    PSO_COGNITIVE_COEF: float = 1.5  # 认知系数(个体最优位置影响)
    PSO_SOCIAL_COEF: float = 1.5  # 社会系数(全局最优位置影响)
    PSO_WEIGHT_CPU: float = 0.25  # CPU资源权重
    PSO_WEIGHT_MEMORY: float = 0.25  # 内存资源权重
    PSO_WEIGHT_BALANCE: float = 0.25  # 负载均衡权重
    PSO_WEIGHT_NETWORK: float = 0.25  # 网络权重
    
    # 模拟退火算法(SA)配置
    SA_INITIAL_TEMP: float = 100.0  # 初始温度
    SA_COOLING_RATE: float = 0.95  # 冷却速率
    SA_MIN_TEMP: float = 0.1  # 最小温度
    SA_ITERATIONS_PER_TEMP: int = 10  # 每个温度下的迭代次数
    SA_WEIGHT_CPU: float = 0.25  # CPU资源权重
    SA_WEIGHT_MEMORY: float = 0.25  # 内存资源权重
    SA_WEIGHT_BALANCE: float = 0.25  # 负载均衡权重
    SA_WEIGHT_NETWORK: float = 0.25  # 网络权重
    
    # 蚁群算法(ACO)配置
    ACO_ANTS_COUNT: int = 20  # 蚂蚁数量
    ACO_ITERATIONS: int = 50  # 迭代次数
    ACO_ALPHA: float = 1.0  # 信息素重要程度系数
    ACO_BETA: float = 2.0  # 启发函数重要程度系数
    ACO_RHO: float = 0.5  # 信息素挥发系数
    ACO_Q: float = 100.0  # 信息素增加强度系数
    ACO_WEIGHT_CPU: float = 0.25  # CPU资源权重
    ACO_WEIGHT_MEMORY: float = 0.25  # 内存资源权重
    ACO_WEIGHT_BALANCE: float = 0.25  # 负载均衡权重
    ACO_WEIGHT_NETWORK: float = 0.25  # 网络权重
    
    # 强化学习配置
    ENABLE_RL: bool = True  # 是否启用强化学习
    RL_ALGORITHM: str = "dqn"  # 强化学习算法: dqn, ppo, a2c
    RL_LEARNING_RATE: float = 0.001  # 学习率
    RL_GAMMA: float = 0.99  # 折扣因子
    RL_EPSILON: float = 0.1  # 探索率
    RL_EPSILON_MIN: float = 0.01  # 最小探索率
    RL_EPSILON_DECAY: float = 0.995  # 探索率衰减系数
    RL_BATCH_SIZE: int = 32  # 批大小
    RL_MEMORY_SIZE: int = 10000  # 经验回放缓冲区大小
    RL_TARGET_UPDATE_FREQ: int = 100  # 目标网络更新频率
    RL_MODEL_SAVE_INTERVAL: int = 1000  # 模型保存间隔(调度次数)
    RL_MODEL_PATH: str = "models/rl"  # 模型保存路径
    RL_HYBRIDIZATION_WEIGHT: float = 0.5  # 强化学习和遗传算法混合权重(0-1)
    
    # 算力量化配置
    ENABLE_COMPUTING_POWER: bool = True
    ENABLE_COMPUTING_POWER_FILTER: bool = True
    COMPUTING_ROUTER_HOST: str = "127.0.0.1"
    COMPUTING_ROUTER_PORT: int = 8888
    COMPUTING_POWER_FALLBACK: bool = True
    
    # 确定性路由软件配置 - 用于业务算力量化
    ROUTER_HOST: str = "127.0.0.1"  # 确定性路由软件的主机地址
    ROUTER_PORT: int = 8889  # 确定性路由软件的端口
    UDP_EXPONENTIAL_BACKOFF: bool = True  # 是否使用指数退避算法
    
    # 算力权重配置
    WEIGHT_CPU_POWER: float = 0.5
    WEIGHT_GPU_POWER: float = 0.3
    WEIGHT_FPGA_POWER: float = 0.2
    
    # 默认算力配置
    DEFAULT_CPU_POWER_RATIO: float = 0.8
    DEFAULT_GPU_TFLOPS: float = 5.0
    DEFAULT_FPGA_TFLOPS: float = 2.0
    
    # 算力信息上报服务配置
    UDP_SERVICE_HOST: str = "0.0.0.0"
    UDP_SERVICE_PORT: int = 9002  # 修改为9002，避免与其他端口冲突
    UDP_REPORT_TIMEOUT: float = 5.0  # 等待算力信息上报超时时间（秒）
    
    # 算力度量配置
    COMPUTING_POWER_ENABLED: bool = True  # 是否启用算力度量功能
    COMPUTING_POWER_CLEANUP_INTERVAL: int = 300  # 清理过期算力信息的间隔（秒）
    COMPUTING_POWER_EXPIRY_TIME: int = 600  # 算力信息过期时间（秒）
    
    # 调度方案存储配置
    PLAN_EXPIRATION_TIME: int = 300  # 调度方案过期时间（秒）
    
    model_config = ConfigDict(
        env_file=".env",
        extra="ignore"  # 允许额外的字段
    )


# 创建全局设置实例
settings = Settings()

# 导出设置
__all__ = ["settings"] 