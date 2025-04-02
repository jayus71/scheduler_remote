"""资源解析工具"""
from loguru import logger


def parse_resource_value(value: str) -> int:
    """解析资源值（CPU cores或内存MB）
    
    Args:
        value: 资源值字符串，例如 "200m"（CPU）或 "256Mi"（内存）
        
    Returns:
        int: 标准化后的资源值（CPU核数*1000或内存MB）
    """
    if not value:
        return 0

    # 移除可能的空格
    value = value.strip()

    # CPU资源处理
    if value.endswith('m'):
        return int(value[:-1])

    # 内存资源处理
    value_upper = value.upper()

    # 处理内存单位
    memory_units = {
        'KI': lambda v: int(v) // 1024,  # KiB to MB
        'MI': lambda v: int(v),  # MiB to MB
        'GI': lambda v: int(v) * 1024,  # GiB to MB
        'K': lambda v: int(v) // 1000,  # KB to MB
        'M': lambda v: int(v),  # MB to MB
        'G': lambda v: int(v) * 1000,  # GB to MB
    }

    # 检查是否匹配任何内存单位
    for unit, converter in memory_units.items():
        if value_upper.endswith(unit):
            return converter(value_upper[:-len(unit)])

    # 如果是纯数字，假设是CPU核数
    if value.isdigit():
        return int(value) * 1000

    # 默认情况，尝试转换为整数
    try:
        return int(value)
    except ValueError:
        logger.warning(f"无法解析资源值: {value}，返回0")
        return 0
