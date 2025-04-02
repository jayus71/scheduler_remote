"""调度方案存储服务模块"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
from loguru import logger


class SchedulingPlanStore:
    """调度方案存储服务类"""
    
    def __init__(self, expiry_time: int = 300):
        """初始化调度方案存储
        
        Args:
            expiry_time: 方案过期时间（秒），默认5分钟
        """
        self._plans: Dict[str, Dict[str, Any]] = {}  # pod_name -> plan
        self._timestamps: Dict[str, datetime] = {}  # pod_name -> timestamp
        self._expiry_time = expiry_time
        self._cleanup_task = None
        
    async def start(self):
        """启动存储服务，包括清理任务"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("调度方案存储服务已启动")
            
    async def stop(self):
        """停止存储服务"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("调度方案存储服务已停止")
            
    async def _cleanup_loop(self):
        """清理过期方案的循环任务"""
        while True:
            try:
                self._cleanup_expired_plans()
                await asyncio.sleep(60)  # 每分钟检查一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"清理过期调度方案时出错: {str(e)}")
                await asyncio.sleep(60)  # 发生错误时等待一分钟后重试
                
    def _cleanup_expired_plans(self):
        """清理过期的调度方案"""
        now = datetime.now()
        expired_pods = [
            pod_name
            for pod_name, timestamp in self._timestamps.items()
            if (now - timestamp).total_seconds() > self._expiry_time
        ]
        
        for pod_name in expired_pods:
            self._plans.pop(pod_name, None)
            self._timestamps.pop(pod_name, None)
            
        if expired_pods:
            logger.info(f"已清理 {len(expired_pods)} 个过期的调度方案")
            
    def store_plan(self, pod_name: str, plan: Dict[str, Any]):
        """存储调度方案
        
        Args:
            pod_name: Pod名称
            plan: 调度方案
        """
        self._plans[pod_name] = plan
        self._timestamps[pod_name] = datetime.now()
        logger.info(f"已存储Pod {pod_name} 的调度方案")
        
    def get_plan(self, pod_name: str) -> Optional[Dict[str, Any]]:
        """获取调度方案
        
        Args:
            pod_name: Pod名称
            
        Returns:
            Optional[Dict[str, Any]]: 调度方案，如果不存在则返回None
        """
        if pod_name not in self._plans:
            return None
            
        # 更新时间戳
        self._timestamps[pod_name] = datetime.now()
        return self._plans[pod_name]
        
    def remove_plan(self, pod_name: str):
        """移除调度方案
        
        Args:
            pod_name: Pod名称
        """
        self._plans.pop(pod_name, None)
        self._timestamps.pop(pod_name, None)
        logger.info(f"已移除Pod {pod_name} 的调度方案") 