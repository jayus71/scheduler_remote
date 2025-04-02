"""算力相关的数据模型"""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from enum import IntEnum
from pydantic import validator


class ComputingPowerRequirement(BaseModel):
    """算力需求数据模型"""
    cpu_freq: int = Field(..., description="所需CPU主频，单位GHz")
    cpu_cores: int = Field(..., description="所需CPU核心数")
    gpu_cores: int = Field(0, description="所需CUDA核心数/流处理器数量")
    gpu_freq: int = Field(0, description="所需GPU核心频率，单位GHz")
    fpga_units: int = Field(0, description="所需FPGA逻辑单元数量/可配置逻辑块数量")

    model_config = {
        "json_schema_extra": {
            "example": {
                "cpu_freq": 2,
                "cpu_cores": 4,
                "gpu_cores": 1024,
                "gpu_freq": 1,
                "fpga_units": 100
            }
        }
    }


class QuantifiedComputingPower(BaseModel):
    """量化后的算力数据模型"""
    cpu_power: int = Field(..., description="CPU计算能力，核心数与主频的乘积")
    gpu_power: float = Field(..., description="GPU计算能力，单位TFLOPS")
    fpga_power: float = Field(..., description="FPGA计算能力，单位TFLOPS")

    model_config = {
        "json_schema_extra": {
            "example": {
                "cpu_power": 8,
                "gpu_power": 5.0,
                "fpga_power": 2.0
            }
        }
    }


class NodeComputingPower(BaseModel):
    """节点算力数据模型"""
    name: str = Field(..., description="节点名称")
    cpu_power: int = Field(..., description="CPU计算能力，核心数与主频的乘积")
    gpu_power: float = Field(..., description="GPU计算能力，单位TFLOPS")
    fpga_power: float = Field(..., description="FPGA计算能力，单位TFLOPS")
    cpu_power_available: int = Field(..., description="可用CPU计算能力")
    gpu_power_available: float = Field(..., description="可用GPU计算能力，单位TFLOPS")
    fpga_power_available: float = Field(..., description="可用FPGA计算能力，单位TFLOPS")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "node-1",
                "cpu_power": 16,
                "gpu_power": 10.0,
                "fpga_power": 5.0,
                "cpu_power_available": 8,
                "gpu_power_available": 5.0,
                "fpga_power_available": 2.5
            }
        }
    }


class ComputingPowerFilterResult(BaseModel):
    """算力过滤结果"""
    suitable_nodes: List[str] = Field(default_factory=list, description="适合的节点名称列表")
    unsuitable_nodes: Dict[str, str] = Field(default_factory=dict, description="不适合的节点及原因")

    model_config = {
        "json_schema_extra": {
            "example": {
                "suitable_nodes": ["node-1", "node-3"],
                "unsuitable_nodes": {
                    "node-2": "GPU算力不足: 需要8.0 TFLOPS, 可用5.0 TFLOPS"
                }
            }
        }
    }


class ComputingForceType(IntEnum):
    """算力资源类型枚举"""
    CPU = 0
    GPU = 1
    FPGA = 2


class CPUInformation(BaseModel):
    """CPU信息"""
    core_count: int = Field(..., description="CPU核心数量", ge=1)
    computing_power: float = Field(..., description="CPU计算能力(核心数×主频)")
    load: float = Field(..., description="CPU负载百分比", ge=0, le=100)

    @validator('load')
    def validate_load(cls, v: float) -> float:
        """验证负载百分比"""
        if not 0 <= v <= 100:
            raise ValueError("负载百分比必须在0-100之间")
        return round(v, 2)


class GPUInformation(BaseModel):
    """GPU信息"""
    tflops: float = Field(..., description="GPU算力(TFLOPS)")
    memory: float = Field(..., description="显存大小(GB)")
    load: float = Field(..., description="GPU负载百分比", ge=0, le=100)

    @validator('load')
    def validate_load(cls, v: float) -> float:
        """验证负载百分比"""
        if not 0 <= v <= 100:
            raise ValueError("负载百分比必须在0-100之间")
        return round(v, 2)


class FPGAInformation(BaseModel):
    """FPGA信息"""
    tflops: float = Field(..., description="FPGA算力(TFLOPS)")
    load: float = Field(..., description="FPGA负载百分比", ge=0, le=100)

    @validator('load')
    def validate_load(cls, v: float) -> float:
        """验证负载百分比"""
        if not 0 <= v <= 100:
            raise ValueError("负载百分比必须在0-100之间")
        return round(v, 2)


class NodeComputingForce(BaseModel):
    """节点算力信息"""
    report_id: int = Field(..., description="报告ID", ge=0, lt=2**32)
    host_name: str = Field(..., description="节点ID", max_length=64)
    total_memory: int = Field(..., description="总内存(GB)", ge=0)
    available_memory: int = Field(..., description="可用内存(GB)", ge=0)
    computing_force_type: ComputingForceType = Field(..., description="算力资源类型")
    
    # CPU相关信息
    cpu_information: Optional[List[CPUInformation]] = Field(
        None, 
        description="CPU信息列表，当computing_force_type=CPU时存在"
    )
    
    # GPU相关信息
    gpu_information: Optional[List[GPUInformation]] = Field(
        None, 
        description="GPU信息列表，当computing_force_type=GPU时存在"
    )
    
    # FPGA相关信息
    fpga_information: Optional[List[FPGAInformation]] = Field(
        None, 
        description="FPGA信息列表，当computing_force_type=FPGA时存在"
    )

    @validator('available_memory')
    def validate_available_memory(cls, v: int, values: dict) -> int:
        """验证可用内存不超过总内存"""
        if 'total_memory' in values and v > values['total_memory']:
            raise ValueError("可用内存不能超过总内存")
        return v

    @validator('cpu_information')
    def validate_cpu_info(cls, v: Optional[List[CPUInformation]], values: dict) -> Optional[List[CPUInformation]]:
        """验证CPU信息"""
        if values.get('computing_force_type') == ComputingForceType.CPU and not v:
            raise ValueError("当算力类型为CPU时，必须提供CPU信息")
        return v

    @validator('gpu_information')
    def validate_gpu_info(cls, v: Optional[List[GPUInformation]], values: dict) -> Optional[List[GPUInformation]]:
        """验证GPU信息"""
        if values.get('computing_force_type') == ComputingForceType.GPU and not v:
            raise ValueError("当算力类型为GPU时，必须提供GPU信息")
        return v

    @validator('fpga_information')
    def validate_fpga_info(cls, v: Optional[List[FPGAInformation]], values: dict) -> Optional[List[FPGAInformation]]:
        """验证FPGA信息"""
        if values.get('computing_force_type') == ComputingForceType.FPGA and not v:
            raise ValueError("当算力类型为FPGA时，必须提供FPGA信息")
        return v


class ComputingForceReport(BaseModel):
    """算力信息上报"""
    report_id: int = Field(..., description="报告ID", ge=0, lt=2**32)
    host_number: int = Field(..., description="节点数量", ge=1, le=255)
    nodes: List[NodeComputingForce] = Field(..., description="节点算力信息列表")

    @validator('nodes')
    def validate_nodes_count(cls, v: List[NodeComputingForce], values: dict) -> List[NodeComputingForce]:
        """验证节点数量与host_number一致"""
        if 'host_number' in values and len(v) != values['host_number']:
            raise ValueError("节点列表长度必须与host_number一致")
        return v 