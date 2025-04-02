"""应用状态管理模块"""
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from loguru import logger

from app.core.config import settings
from app.services.network_service import NetworkService
from app.services.network_info_service import NetworkInfoService
from app.services.computing_power_service import ComputingPowerService
from app.services.scheduling_plan_store import SchedulingPlanStore

# 全局状态
_network_service: Optional[NetworkService] = None
_network_info_service: Optional[NetworkInfoService] = None
_computing_power_service: Optional[ComputingPowerService] = None
_scheduling_plan_store: Optional[SchedulingPlanStore] = None


def get_network_info_service() -> Optional[NetworkInfoService]:
    """获取网络信息服务实例"""
    return _network_info_service


def get_computing_power_service() -> Optional[ComputingPowerService]:
    """获取算力服务实例"""
    return _computing_power_service


def get_scheduling_plan_store() -> Optional[SchedulingPlanStore]:
    """获取调度方案存储服务实例"""
    return _scheduling_plan_store


@asynccontextmanager
async def manage_network_info_service():
    """管理网络信息服务的生命周期"""
    global _network_info_service
    try:
        _network_info_service = NetworkInfoService()
        await _network_info_service.start_server()
        logger.info("网络信息服务已启动")
        yield _network_info_service
    finally:
        if _network_info_service:
            await _network_info_service.stop_server()
            _network_info_service = None
            logger.info("网络信息服务已关闭")


@asynccontextmanager
async def manage_computing_power_service():
    """管理算力服务的生命周期"""
    global _computing_power_service
    try:
        _computing_power_service = ComputingPowerService()
        await _computing_power_service.start()
        logger.info("算力服务已启动")
        yield _computing_power_service
    finally:
        if _computing_power_service:
            await _computing_power_service.stop()
            _computing_power_service = None
            logger.info("算力服务已关闭")


@asynccontextmanager
async def manage_scheduling_plan_store():
    """管理调度方案存储服务的生命周期"""
    global _scheduling_plan_store
    try:
        _scheduling_plan_store = SchedulingPlanStore()
        await _scheduling_plan_store.start()
        logger.info("调度方案存储服务已启动")
        yield _scheduling_plan_store
    finally:
        if _scheduling_plan_store:
            await _scheduling_plan_store.stop()
            _scheduling_plan_store = None
            logger.info("调度方案存储服务已关闭")


@asynccontextmanager
async def manage_services():
    """统一管理所有服务的生命周期"""
    global _network_info_service, _computing_power_service, _scheduling_plan_store
    
    try:
        # 直接初始化服务，而不是使用嵌套的上下文管理器
        # 这样可以确保所有服务都在同一个作用域内初始化
        _network_info_service = NetworkInfoService()
        _computing_power_service = ComputingPowerService()
        _scheduling_plan_store = SchedulingPlanStore()
        
        # 启动所有服务
        await _network_info_service.start_server()
        logger.info("网络信息服务已启动")
        
        await _computing_power_service.start()
        logger.info("算力服务已启动")
        
        await _scheduling_plan_store.start()
        logger.info("调度方案存储服务已启动")
        
        # 等待一段时间确保服务完全初始化
        await asyncio.sleep(1)
        
        # 验证服务是否正确初始化
        if _network_info_service is None:
            logger.error("网络信息服务初始化失败")
        else:
            logger.info("网络信息服务初始化成功")
            
        if _computing_power_service is None:
            logger.error("算力服务初始化失败")
        else:
            logger.info("算力服务初始化成功")
            
        if _scheduling_plan_store is None:
            logger.error("调度方案存储服务初始化失败")
        else:
            logger.info("调度方案存储服务初始化成功")
        
        logger.info("所有服务已启动")
        yield
    except Exception as e:
        logger.error(f"服务初始化失败: {str(e)}")
        # 确保服务实例在初始化失败时被正确清理
        _network_info_service = None
        _computing_power_service = None
        _scheduling_plan_store = None
        raise
    finally:
        # 关闭所有服务
        if _scheduling_plan_store:
            try:
                await _scheduling_plan_store.stop()
                logger.info("调度方案存储服务已关闭")
            except Exception as e:
                logger.error(f"关闭调度方案存储服务时出错: {str(e)}")
            finally:
                _scheduling_plan_store = None
            
        if _computing_power_service:
            try:
                await _computing_power_service.stop()
                logger.info("算力服务已关闭")
            except Exception as e:
                logger.error(f"关闭算力服务时出错: {str(e)}")
            finally:
                _computing_power_service = None
            
        if _network_info_service:
            try:
                await _network_info_service.stop_server()
                logger.info("网络信息服务已关闭")
            except Exception as e:
                logger.error(f"关闭网络信息服务时出错: {str(e)}")
            finally:
                _network_info_service = None
            
        logger.info("所有服务已关闭")


async def init_app_state():
    """
    初始化应用状态
    
    启动必要的服务和后台任务
    """
    global _network_service, _external_k8s_service
    
    # 初始化网络服务
    try:
        _network_service = NetworkService(
            host=settings.NETWORK_SERVICE_HOST,
            port=settings.NETWORK_SERVICE_PORT
        )
        # 启动UDP服务器
        asyncio.create_task(_network_service.start_server())
        logger.info(f"网络服务已初始化，监听地址: {settings.NETWORK_SERVICE_HOST}:{settings.NETWORK_SERVICE_PORT}")
    except Exception as e:
        logger.error(f"初始化网络服务失败: {str(e)}")
        _network_service = None
        


async def shutdown_app_state():
    """关闭应用状态，清理资源"""
    global _network_service
    
    # 关闭网络服务
    if _network_service:
        await _network_service.stop()
        logger.info("网络服务已关闭")



def get_network_service() -> Optional[NetworkService]:
    """
    获取网络服务实例
    
    Returns:
        Optional[NetworkService]: 网络服务实例，如果未初始化则返回None
    """
    return _network_service
