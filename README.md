# 星载微服务协同系统

星载微服务协同系统是一个专为星载计算环境设计的微服务调度系统，能够根据网络状况和计算资源动态调整服务部署策略。

## 功能特点

- 基于Kubernetes的微服务调度
- 网络状况感知的服务部署
- 计算资源优化分配
- 支持多种连接方式（ServiceAccount或本地kubeconfig）
- 支持离线部署模式
- 强化学习与遗传算法混合优化的智能调度

## 系统架构

系统由以下主要组件构成：

- **HTTP API服务**：提供RESTful API接口，用于接收调度请求和查询系统状态
- **网络信息服务**：通过UDP协议收集网络状况信息
- **算力信息服务**：通过UDP协议收集计算资源信息
- **调度优化器**：基于遗传算法与强化学习的混合节点评分系统
- **过滤服务**：使用整数规划算法进行节点筛选

### 调度优化器

调度优化器是系统的核心组件，负责为Pod选择最合适的节点。目前支持以下调度算法：

- **遗传算法**：通过模拟自然选择和基因重组过程优化节点选择
- **粒子群优化(PSO)**：模拟鸟群觅食行为，通过个体与群体最优解协作寻找全局最优解
- **模拟退火(SA)**：模拟金属退火过程，以一定概率接受较差解以跳出局部最优，随着"温度"下降逐渐收敛到全局最优
- **蚁群算法(ACO)**：模拟蚂蚁觅食过程中的信息素通信机制，通过正反馈不断优化路径选择
- **强化学习**：使用深度Q网络(DQN)算法，通过经验学习持续优化调度决策
- **混合模式**：结合算法和强化学习的优点，通过可配置的权重进行混合决策

## 部署指南

### 前提条件

- Docker 20.10+
- Kubernetes集群（可选）
- Python 3.11+（仅开发环境需要）
- CUDA支持（可选，用于GPU加速强化学习训练）

### 环境变量配置

系统支持通过.env文件配置以下环境变量：

| 环境变量                    | 描述                          | 默认值      |
| --------------------------- | ----------------------------- | ----------- |
| `USE_SERVICE_ACCOUNT`     | 是否使用ServiceAccount连接K8s | `true`    |
| `KUBECONFIG`              | kubeconfig文件路径            | -           |
| `UDP_SERVICE_HOST`        | UDP服务监听地址               | `0.0.0.0` |
| `UDP_SERVICE_PORT`        | UDP服务监听端口               | `9001`    |
| `COMPUTING_POWER_PORT`    | 算力信息服务端口              | `9002`    |
| `LOG_LEVEL`               | 日志级别                      | `INFO`    |
| `ENABLE_RL`               | 是否启用强化学习              | `true`    |
| `RL_ALGORITHM`            | 强化学习算法                  | `dqn`     |
| `RL_HYBRIDIZATION_WEIGHT` | 强化学习混合权重              | `0.5`     |
| `OPTIMIZATION_METHOD`     | 优化方法选择(genetic, pso, sa, aco) | `genetic` |
| `PSO_SWARM_SIZE`          | PSO算法粒子群大小             | `30`      |
| `PSO_MAX_ITERATIONS`      | PSO算法最大迭代次数           | `50`      |
| `PSO_INERTIA_WEIGHT`      | PSO算法惯性权重               | `0.7`     |
| `PSO_COGNITIVE_COEF`      | PSO算法认知系数               | `1.5`     |
| `PSO_SOCIAL_COEF`         | PSO算法社会系数               | `1.5`     |
| `SA_INITIAL_TEMP`          | SA算法初始温度                | `100.0`   |
| `SA_COOLING_RATE`          | SA算法冷却速率                | `0.95`    |
| `SA_MIN_TEMP`              | SA算法最小温度                | `0.1`     |
| `SA_ITERATIONS_PER_TEMP`   | SA算法每温度迭代次数          | `10`      |
| `ACO_ANTS_COUNT`           | ACO蚂蚁数量                   | `20`      |
| `ACO_ITERATIONS`           | ACO迭代次数                   | `50`      |
| `ACO_ALPHA`                | ACO信息素重要程度系数         | `1.0`     |
| `ACO_BETA`                 | ACO启发函数重要程度系数       | `2.0`     |
| `ACO_RHO`                  | ACO信息素挥发系数             | `0.5`     |
| `ACO_Q`                    | ACO信息素增加强度系数         | `100.0`   |

### 创建.env文件

在项目根目录创建.env文件：

```bash
# 创建基本的.env文件
cat > .env << EOF
# 基本配置
APP_NAME=edge-scheduler
LOG_LEVEL=INFO

# Kubernetes配置
USE_SERVICE_ACCOUNT=false
# 启用K8s模拟模式，无需真实的Kubernetes集群
MOCK_K8S=true

# UDP服务配置
UDP_SERVICE_HOST=0.0.0.0
UDP_SERVICE_PORT=9001
COMPUTING_POWER_PORT=9002

# 强化学习配置
ENABLE_RL=true
RL_ALGORITHM=dqn
RL_HYBRIDIZATION_WEIGHT=0.5

# 优化算法配置
OPTIMIZATION_METHOD=genetic  # 可选：genetic, pso, sa
# PSO算法参数
PSO_SWARM_SIZE=30
PSO_MAX_ITERATIONS=50
PSO_INERTIA_WEIGHT=0.7
PSO_COGNITIVE_COEF=1.5
PSO_SOCIAL_COEF=1.5

# SA算法参数
SA_INITIAL_TEMP=100.0
SA_COOLING_RATE=0.95
SA_MIN_TEMP=0.1
SA_ITERATIONS_PER_TEMP=10

# ACO算法参数
ACO_ANTS_COUNT=20
ACO_ITERATIONS=50
ACO_ALPHA=1.0
ACO_BETA=2.0
ACO_RHO=0.5
ACO_Q=100.0
EOF
```

### 使用Docker部署

#### 构建镜像

```bash
docker build -t edge-scheduler:latest .
```

#### 启动容器

```bash
# 基本启动命令（使用.env文件配置）
docker run -d --name edge-scheduler \
  -p 8000:8000 \
  -p 9001:9001/udp \
  -p 9002:9002/udp \
  --env-file .env \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  edge-scheduler:latest
```

如果需要使用本地kubeconfig文件，可以添加以下挂载：

```bash
# 挂载kubeconfig文件
-v $(pwd)/kubeconfig.yaml:/app/kubeconfig.yaml:ro
```

### 连接Kubernetes集群

系统支持三种方式连接Kubernetes集群：

#### 1. 使用模拟模式（无需真实集群）

如果您没有可用的Kubernetes集群或只需要进行功能测试，可以启用模拟模式：

1. 在.env文件中设置：

```
MOCK_K8S=true
```

2. 此模式下，系统将模拟Kubernetes API响应，无需真实的集群连接。

#### 2. 使用ServiceAccount（推荐用于生产环境）

1. 应用ServiceAccount配置：

```bash
kubectl apply -f k8s/service-account.yaml
```

2. 在.env文件中设置：

```
USE_SERVICE_ACCOUNT=true
MOCK_K8S=false
```

#### 3. 使用本地kubeconfig文件

1. 准备kubeconfig文件并放置在项目根目录
2. 在.env文件中设置：

```
USE_SERVICE_ACCOUNT=false
MOCK_K8S=false
```

3. 启动容器时挂载该文件：

```bash
-v $(pwd)/kubeconfig.yaml:/app/kubeconfig.yaml:ro
```

系统会按以下顺序查找kubeconfig文件：

- `/app/kubeconfig.yaml`（容器内路径）
- `~/.kube/config`（容器内用户目录）
- 环境变量 `KUBECONFIG`指定的路径

### 健康检查

系统提供以下健康检查端点：

- `GET /health`：检查系统整体健康状态
- `GET /health/kubernetes`：检查Kubernetes连接状态
- `GET /health/udp`：检查UDP服务状态

示例：

```bash
curl http://localhost:8000/health
```

## API文档

启动服务后，可通过以下地址访问API文档：

```
http://localhost:8000/docs
```

## 强化学习功能

系统支持基于深度强化学习的智能调度优化。强化学习代理通过与环境交互，不断学习和优化调度决策。

### 主要特性

- **DQN算法**：使用深度Q网络进行状态-动作价值估计
- **经验回放**：存储历史调度经验，实现离线批量学习
- **持续学习**：调度系统在运行过程中不断优化调度策略
- **混合决策**：可配置的遗传算法与强化学习混合权重

### 配置参数

| 参数                        | 描述                       | 默认值    |
| --------------------------- | -------------------------- | --------- |
| `ENABLE_RL`               | 是否启用强化学习           | `true`  |
| `RL_ALGORITHM`            | 强化学习算法类型           | `dqn`   |
| `RL_LEARNING_RATE`        | 学习率                     | `0.001` |
| `RL_GAMMA`                | 折扣因子                   | `0.99`  |
| `RL_EPSILON`              | 探索率                     | `0.1`   |
| `RL_BATCH_SIZE`           | 批大小                     | `32`    |
| `RL_HYBRIDIZATION_WEIGHT` | 强化学习与遗传算法混合权重 | `0.5`   |

### 模型保存与加载

强化学习模型会自动保存在 `models/rl`目录，并在下次启动时自动加载最新模型。

## 离线部署

详细的离线部署指南请参考[离线部署文档](offline_deployment.md)。

## 开发指南

### 开发环境设置

1. 克隆代码库：

```bash
git clone https://github.com/yourusername/edge-scheduler.git
cd edge-scheduler
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 创建.env文件：

```bash
cat > .env << EOF
APP_NAME=edge-scheduler
LOG_LEVEL=DEBUG
USE_SERVICE_ACCOUNT=false
# 开发环境建议启用模拟模式
MOCK_K8S=true
UDP_SERVICE_HOST=0.0.0.0
UDP_SERVICE_PORT=9001
COMPUTING_POWER_PORT=9002

# 强化学习配置
ENABLE_RL=true
RL_ALGORITHM=dqn
RL_LEARNING_RATE=0.001
RL_GAMMA=0.99
RL_EPSILON=0.1
RL_HYBRIDIZATION_WEIGHT=0.5
EOF
```

4. 运行开发服务器：

```bash
uvicorn app.main:app --reload
```

### 运行测试

```bash
pytest
```

## 故障排除

### 常见问题

1. **无法连接到Kubernetes集群**

   - 检查kubeconfig.yaml文件是否存在且不为空
   - 确认.env文件中的 `MOCK_K8S`和 `USE_SERVICE_ACCOUNT`设置正确
   - 如果不需要真实集群连接，设置 `MOCK_K8S=true`
2. **kubeconfig文件加载失败**

   - 确保kubeconfig.yaml文件格式正确
   - 检查文件权限是否正确
   - 尝试使用模拟模式：`MOCK_K8S=true`
3. **强化学习模型加载失败**

   - 确认 `models/rl`目录存在且权限正确
   - 检查是否安装了所有必要的依赖库（torch、numpy、joblib等）
   - 如遇问题可尝试禁用强化学习：`ENABLE_RL=false`

## 开发与扩展

### 添加新的强化学习算法

1. 在 `app/services/rl_agent.py`文件中添加新的算法实现
2. 更新配置处理逻辑以支持新算法
3. 在 `.env`文件中设置 `RL_ALGORITHM`为新算法名称

## 许可证

[MIT](LICENSE)



