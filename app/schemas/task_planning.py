"""联合任务规划数据模型"""
from typing import List, Dict, Optional, Any, Annotated
from pydantic import BaseModel, Field, field_validator, model_validator
import re


class Task(BaseModel):
    """单个任务模型"""
    task_id: Annotated[str, Field(
        description="任务ID，唯一标识符，必须符合Kubernetes命名规范",
        min_length=1,
        max_length=63,
        pattern=r'^[a-z0-9]([-a-z0-9]*[a-z0-9])?$'
    )]
    name: Annotated[str, Field(
        description="任务名称",
        min_length=1,
        max_length=63
    )]
    image: str = Field(description="容器镜像")
    dependencies: List[str] = Field(default=[], description="依赖的任务ID列表")
    cpu: str = Field(default="100m", description="CPU请求量，支持m（毫核）或直接指定核心数")
    memory: str = Field(default="128Mi", description="内存请求量，支持Mi、Gi等单位")
    commands: Optional[List[str]] = Field(default=None, description="容器启动命令")
    args: Optional[List[str]] = Field(default=None, description="命令参数")
    env_vars: Optional[Dict[str, str]] = Field(default=None, description="环境变量")
    namespace: str = Field(default="default", description="部署的命名空间")
    
    @field_validator('dependencies')
    @classmethod
    def validate_dependencies(cls, v, info):
        """验证依赖关系"""
        if 'task_id' in info.data and info.data['task_id'] in v:
            raise ValueError('任务不能依赖自身')
        return v
    
    @field_validator('cpu')
    @classmethod
    def validate_cpu(cls, v):
        """验证CPU格式"""
        if not re.match(r'^\d+m$|^\d+(\.\d+)?$', v):
            raise ValueError('CPU格式无效，应为数字后跟m（如100m）或直接指定核心数（如0.5）')
        return v
    
    @field_validator('memory')
    @classmethod
    def validate_memory(cls, v):
        """验证内存格式"""
        if not re.match(r'^\d+[KMGT]i?$', v):
            raise ValueError('内存格式无效，应为数字后跟Ki、Mi、Gi、Ti（如128Mi）')
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "task_id": "task-1",
                "name": "data-preparation",
                "image": "python:3.9",
                "dependencies": [],
                "cpu": "200m",
                "memory": "256Mi",
                "commands": ["python", "-c"],
                "args": ["print('Data preparation completed')"],
                "env_vars": {"DEBUG": "true"},
                "namespace": "ml-tasks"
            }
        }
    }


class JointTasksPlan(BaseModel):
    """联合任务规划模型"""
    plan_name: Annotated[str, Field(
        description="规划名称，必须符合Kubernetes命名规范",
        min_length=1,
        max_length=63,
        pattern=r'^[a-z0-9]([-a-z0-9]*[a-z0-9])?$'
    )]
    description: Optional[str] = Field(default=None, description="规划描述")
    tasks: List[Task] = Field(..., min_items=1, description="任务列表，至少包含一个任务")
    
    @model_validator(mode='after')
    def validate_tasks(self):
        """验证任务列表"""
        task_ids = set()
        modified_tasks = {}
        
        # 第一次遍历：收集所有任务ID
        for task in self.tasks:
            if task.task_id in task_ids:
                raise ValueError(f'任务ID重复: {task.task_id}')
            task_ids.add(task.task_id)
        
        # 第二次遍历：验证和清理依赖
        for task in self.tasks:
            # 检查是否有无效依赖
            invalid_deps = []
            for dep in task.dependencies:
                if dep not in task_ids:
                    invalid_deps.append(dep)
            
            # 如果有无效依赖，自动清理
            if invalid_deps:
                modified_tasks[task.task_id] = invalid_deps
                task.dependencies = [dep for dep in task.dependencies if dep in task_ids]
        
        return self
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "plan_name": "ml-pipeline",
                "description": "A machine learning pipeline with data prep, training, and evaluation",
                "tasks": [
                    {
                        "task_id": "task-1",
                        "name": "data-preparation",
                        "image": "python:3.9",
                        "dependencies": [],
                        "cpu": "200m",
                        "memory": "256Mi"
                    },
                    {
                        "task_id": "task-2",
                        "name": "model-training",
                        "image": "tensorflow/tensorflow:latest",
                        "dependencies": ["task-1"],
                        "cpu": "500m",
                        "memory": "1Gi"
                    }
                ]
            }
        }
    }


class TaskPlanningResponse(BaseModel):
    """任务规划响应模型"""
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="响应消息")
    plan_name: Optional[str] = Field(default=None, description="规划名称")
    serialized_tasks: Optional[List[str]] = Field(default=None, description="拓扑排序后的任务ID序列，表示任务的执行顺序")
    deployed_tasks: Optional[List[str]] = Field(default=None, description="按拓扑排序顺序成功部署的任务ID列表")
    failed_tasks: Optional[Dict[str, str]] = Field(default=None, description="部署失败的任务及原因")
    modified_dependencies: Optional[Dict[str, List[str]]] = Field(default=None, description="被移除的无效依赖项")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "message": "所有任务成功部署，按顺序执行: task-1, task-2, task-3, task-4",
                "plan_name": "ml-pipeline",
                "serialized_tasks": ["task-1", "task-2", "task-3", "task-4"],
                "deployed_tasks": ["task-1", "task-2", "task-3", "task-4"],
                "failed_tasks": {},
                "modified_dependencies": {"task-1": ["test-1", "test-2"]}
            }
        }
    } 