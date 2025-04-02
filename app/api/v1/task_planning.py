"""联合任务规划API路由模块"""
import os
import json
from typing import List, Dict, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Body, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.schemas.task_planning import JointTasksPlan, TaskPlanningResponse
from app.services.task_planning_service import TaskPlanningService

router = APIRouter(tags=["task-planning"], prefix="/task-planning")
task_planning_service = TaskPlanningService()

# 捕获和处理所有可能的CORS问题
@router.options("/{path:path}")
async def options_handler(path: str):
    """处理CORS预检请求"""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
    )


@router.post(
    "/upload",
    response_model=TaskPlanningResponse,
    summary="上传联合任务规划文件",
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"description": "无效的文件格式或内容"},
        500: {"description": "服务器内部错误"}
    }
)
async def upload_tasks_plan(
    file: UploadFile = File(..., description="联合任务规划文件（JSON或YAML格式）"),
    deploy: bool = Form(True, description="是否立即部署任务")
) -> TaskPlanningResponse:
    """
    上传联合任务规划文件并处理
    
    接收JSON或YAML格式的联合任务规划文件，解析后进行拓扑排序，并可选择立即部署任务。
    任务将按照拓扑排序结果（依赖顺序）部署，响应中的serialized_tasks字段表示任务的执行顺序。
    
    Args:
        file: 上传的联合任务规划文件
        deploy: 是否立即部署任务
        
    Returns:
        TaskPlanningResponse: 任务规划响应，包含任务执行顺序和部署状态
        
    Raises:
        HTTPException: 当文件格式不支持或解析失败时
    """
    try:
        logger.info(f"接收到联合任务规划文件: {file.filename}, 部署选项: {deploy}")
        
        # 获取文件扩展名
        _, file_extension = os.path.splitext(file.filename)
        if file_extension.lower() not in ['.json', '.yaml', '.yml']:
            logger.error(f"不支持的文件格式: {file_extension}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "message": f"不支持的文件格式: {file_extension}，仅支持 .json, .yaml, .yml"},
                headers={"Access-Control-Allow-Origin": "*"}
            )
        
        # 读取文件内容
        try:
            file_content = await file.read()
        except Exception as e:
            logger.error(f"读取文件内容失败: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "message": f"读取文件内容失败: {str(e)}"},
                headers={"Access-Control-Allow-Origin": "*"}
            )
        
        # 解析文件
        try:
            plan = await task_planning_service.parse_tasks_file(file_content, file_extension)
            logger.info(f"成功解析联合任务规划: {plan.plan_name}, 任务数量: {len(plan.tasks)}")
            
            # 检查是否有被修正的依赖
            modified_dependencies = getattr(plan, 'modified_dependencies', {})
            if modified_dependencies:
                logger.info(f"任务规划中有 {len(modified_dependencies)} 个任务的依赖关系被修正")
        except ValueError as e:
            logger.error(f"解析任务规划文件失败: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "message": str(e)},
                headers={"Access-Control-Allow-Origin": "*"}
            )
        
        # 如果不需要立即部署，只返回拓扑排序结果
        if not deploy:
            try:
                sorted_tasks, _ = await task_planning_service.topological_sort(plan)
                response = TaskPlanningResponse(
                    success=True,
                    message="联合任务规划成功解析，但未部署",
                    plan_name=plan.plan_name,
                    serialized_tasks=sorted_tasks,
                    modified_dependencies=modified_dependencies if modified_dependencies else None
                )
                return JSONResponse(
                    status_code=status.HTTP_200_OK,
                    content=response.dict(exclude_none=True),
                    headers={"Access-Control-Allow-Origin": "*"}
                )
            except ValueError as e:
                logger.error(f"拓扑排序失败: {str(e)}")
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"success": False, "message": str(e)},
                    headers={"Access-Control-Allow-Origin": "*"}
                )
        
        # 处理任务规划（拓扑排序和部署）
        try:
            response = await task_planning_service.process_tasks_plan(plan)
            return JSONResponse(
                status_code=status.HTTP_201_CREATED,
                content=response.dict(exclude_none=True),
                headers={"Access-Control-Allow-Origin": "*"}
            )
        except Exception as e:
            logger.error(f"处理任务规划失败: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"success": False, "message": f"处理任务规划失败: {str(e)}"},
                headers={"Access-Control-Allow-Origin": "*"}
            )
            
    except Exception as e:
        logger.error(f"处理任务规划文件时发生未知错误: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "message": f"处理任务规划文件时发生未知错误: {str(e)}"},
            headers={"Access-Control-Allow-Origin": "*"}
        )


@router.post(
    "/process",
    response_model=TaskPlanningResponse,
    summary="处理联合任务规划",
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"description": "无效的任务规划"},
        500: {"description": "服务器内部错误"}
    }
)
async def process_tasks_plan(
    plan: JointTasksPlan = Body(..., description="联合任务规划")
) -> JSONResponse:
    """
    处理联合任务规划
    
    接收联合任务规划JSON数据，进行拓扑排序并按顺序部署任务。
    系统会检测任务间的依赖关系，形成有向无环图，并按照拓扑排序顺序部署任务。
    如果检测到循环依赖，将会返回错误信息并指出具体的循环路径。
    
    Args:
        plan: 联合任务规划
        
    Returns:
        TaskPlanningResponse: 任务规划响应，包含任务执行顺序和部署状态
        
    Raises:
        HTTPException: 当任务规划处理失败时（如存在循环依赖）
    """
    try:
        logger.info(f"接收到联合任务规划: {plan.plan_name}, 任务数量: {len(plan.tasks)}")
        
        # 处理任务规划（拓扑排序和部署）
        try:
            response = await task_planning_service.process_tasks_plan(plan)
            return JSONResponse(
                status_code=status.HTTP_201_CREATED,
                content=response.dict(exclude_none=True),
                headers={"Access-Control-Allow-Origin": "*"}
            )
        except ValueError as e:
            logger.error(f"处理任务规划失败: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"success": False, "message": str(e)},
                headers={"Access-Control-Allow-Origin": "*"}
            )
            
    except Exception as e:
        logger.error(f"处理任务规划时发生未知错误: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "message": f"处理任务规划时发生未知错误: {str(e)}"},
            headers={"Access-Control-Allow-Origin": "*"}
        )


@router.get(
    "/example",
    summary="获取联合任务规划示例",
    response_description="返回一个示例联合任务规划，展示正确的格式和结构"
)
async def get_tasks_plan_example() -> Dict:
    """
    获取联合任务规划示例
    
    返回一个示例联合任务规划，可用于了解格式
    
    Returns:
        Dict: 示例联合任务规划
    """
    example = {
        "plan_name": "ml-pipeline",
        "description": "A machine learning pipeline with data preparation, training, and evaluation",
        "tasks": [
            {
                "task_id": "task-1",
                "name": "data-preparation",
                "image": "python:3.9",
                "dependencies": [],
                "cpu": "200m",
                "memory": "256Mi",
                "commands": ["python", "-c"],
                "args": ["print('Data preparation completed')"],
                "env_vars": {"DEBUG": "true"},
                "namespace": "default"
            },
            {
                "task_id": "task-2",
                "name": "model-training",
                "image": "tensorflow/tensorflow:latest",
                "dependencies": ["task-1"],
                "cpu": "500m",
                "memory": "1Gi",
                "commands": ["python", "-c"],
                "args": ["print('Model training completed')"],
                "namespace": "default"
            },
            {
                "task_id": "task-3",
                "name": "model-evaluation",
                "image": "python:3.9",
                "dependencies": ["task-2"],
                "cpu": "300m",
                "memory": "512Mi",
                "commands": ["python", "-c"],
                "args": ["print('Model evaluation completed')"],
                "namespace": "default"
            },
            {
                "task_id": "task-4",
                "name": "model-deployment",
                "image": "nginx:latest",
                "dependencies": ["task-3"],
                "cpu": "100m",
                "memory": "128Mi",
                "env_vars": {"ENVIRONMENT": "production"},
                "namespace": "default"
            }
        ]
    }
    
    return example 