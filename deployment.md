# Edge Scheduler 部署指南

本文档提供在现有 Kubernetes 集群中部署 Edge Scheduler 的详细步骤，重点关注环境变量配置。适用于将 Edge Scheduler 部署到新的 Kubernetes 环境中。

## 新环境对接概述

将 Edge Scheduler 部署到新的 Kubernetes 集群需要完成以下关键步骤：

1. **环境评估**：确认新环境的兼容性和准备工作
2. **镜像准备**：将 Docker 镜像迁移到新环境
3. **配置适配**：调整环境变量以适配新环境
4. **权限设置**：确保应用具有必要的访问权限
5. **部署验证**：确认应用在新环境中正常运行

详细的对接步骤请参考文档末尾的"新环境对接指南"章节。

## 前置条件

- 已有可用的 Kubernetes 集群（1.20.0 或更高版本）
- kubectl 命令行工具已配置并可访问目标集群
- 已构建的 edge-scheduler Docker 镜像

## 部署步骤

### 1. 准备镜像

有两种方式准备镜像：

#### 方式一：推送到镜像仓库

```bash
# 为镜像添加标签，指向目标仓库
docker tag edge-scheduler:latest <your-registry>/edge-scheduler:latest

# 推送到镜像仓库
docker push <your-registry>/edge-scheduler:latest
```

#### 方式二：导出并导入镜像（适用于无法访问外部仓库的环境）

```bash
# 导出镜像为tar文件
docker save -o edge-scheduler.tar edge-scheduler:latest

# 将tar文件传输到目标环境

# 在目标环境导入镜像
docker load -i edge-scheduler.tar
```

### 2. 创建命名空间

```bash
kubectl create namespace edge-scheduler
```

### 3. 配置环境变量

Edge Scheduler 的环境变量配置是部署的关键。有两种方式配置环境变量：

#### 方式一：使用 ConfigMap（推荐）

1. 创建 ConfigMap 配置文件 `edge-scheduler-configmap.yaml`：

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: edge-scheduler-config
  namespace: edge-scheduler
data:
  # Kubernetes配置
  USE_SERVICE_ACCOUNT: "true"
  KUBE_NAMESPACE: "edge-scheduler"
  
  # 应用配置
  APP_NAME: "edge-scheduler"
  APP_VERSION: "1.0.0"
  DEBUG: "false"
  LOG_LEVEL: "INFO"
  
  # API配置
  API_V1_PREFIX: "/v1"
  HOST: "0.0.0.0"
  PORT: "8000"
  
  # 调度策略配置
  SCHEDULER_FILTER_TIMEOUT: "5"
  MAX_PODS_PER_NODE: "110"
  ENABLE_GPU_SCHEDULING: "false"
  
  # 优化器配置
  ENABLE_OPTIMIZER: "true"
  WEIGHT_CPU: "0.6"
  WEIGHT_MEMORY: "0.4"
  OPTIMIZER_ALGORITHM: "default"
  
  # 遗传算法配置
  GA_POPULATION_SIZE: "50"
  GA_GENERATIONS: "100"
  GA_MUTATION_RATE: "0.1"
  GA_WEIGHT_CPU: "0.5"
  GA_WEIGHT_MEMORY: "0.3"
  GA_WEIGHT_BALANCE: "0.2"

  # 联合任务规划配置
  TASK_PLANNING_ENABLED: "true"
  TASK_PLANNING_MAX_TASKS: "100"
  TASK_PLANNING_DEFAULT_NAMESPACE: "default"
  TASK_PLANNING_DEPLOY_TIMEOUT: "300"
  TASK_PLANNING_RETRY_COUNT: "3"
  TASK_PLANNING_RETRY_INTERVAL: "5"
```

2. 应用 ConfigMap：

```bash
kubectl apply -f edge-scheduler-configmap.yaml
```

#### 方式二：使用 .env 文件作为 ConfigMap

如果希望保持与原始 .env 文件格式一致：

1. 创建 ConfigMap 配置文件 `edge-scheduler-env-configmap.yaml`：

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: edge-scheduler-env-config
  namespace: edge-scheduler
data:
  .env: |
    # Kubernetes配置
    USE_SERVICE_ACCOUNT=true
    KUBE_NAMESPACE=edge-scheduler
    
    # 应用配置
    APP_NAME=edge-scheduler
    APP_VERSION=1.0.0
    DEBUG=false
    LOG_LEVEL=INFO
    
    # API配置
    API_V1_PREFIX=/v1
    HOST=0.0.0.0
    PORT=8000
    
    # 调度策略配置
    SCHEDULER_FILTER_TIMEOUT=5
    MAX_PODS_PER_NODE=110
    ENABLE_GPU_SCHEDULING=false
    
    # 监控配置 - 需要根据新环境修改
    ENABLE_METRICS=true
    METRICS_PORT=9090
    
    # 监控API配置 - 需要根据新环境修改
    MONITOR_API_URL=http://<your-prometheus-or-thanos-service>:9091
    MONITOR_API_VERSION=v1
    MONITOR_REQUEST_TIMEOUT=5
    MONITOR_METRICS_WINDOW=5
    
    # 优化器配置
    ENABLE_OPTIMIZER=true
    WEIGHT_CPU=0.6
    WEIGHT_MEMORY=0.4
    OPTIMIZER_ALGORITHM=default
    
    # 遗传算法配置
    GA_POPULATION_SIZE=50
    GA_GENERATIONS=100
    GA_MUTATION_RATE=0.1
    GA_WEIGHT_CPU=0.5
    GA_WEIGHT_MEMORY=0.3
    GA_WEIGHT_BALANCE=0.2

    # 联合任务规划配置
    TASK_PLANNING_ENABLED=true
    TASK_PLANNING_MAX_TASKS=100
    TASK_PLANNING_DEFAULT_NAMESPACE=default
    TASK_PLANNING_DEPLOY_TIMEOUT=300
    TASK_PLANNING_RETRY_COUNT=3
    TASK_PLANNING_RETRY_INTERVAL=5
```

2. 应用 ConfigMap：

```bash
kubectl apply -f edge-scheduler-env-configmap.yaml
```

### 4. 创建 ServiceAccount 和 RBAC 权限

创建 `edge-scheduler-rbac.yaml` 文件：

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: edge-scheduler
  namespace: edge-scheduler
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: edge-scheduler
subjects:
  - kind: ServiceAccount
    name: edge-scheduler
    namespace: edge-scheduler
roleRef:
  kind: ClusterRole
  name: cluster-admin
  apiGroup: rbac.authorization.k8s.io
```

应用 RBAC 配置：

```bash
kubectl apply -f edge-scheduler-rbac.yaml
```

### 5. 创建 Deployment

根据使用的环境变量配置方式，选择对应的 Deployment 配置：

#### 使用 ConfigMap 键值对方式

创建 `edge-scheduler-deployment.yaml` 文件：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: edge-scheduler
  namespace: edge-scheduler
  labels:
    app: edge-scheduler
    component: scheduler-extender
    tier: control-plane
spec:
  replicas: 1
  selector:
    matchLabels:
      app: edge-scheduler
      component: scheduler-extender
      tier: control-plane
  template:
    metadata:
      labels:
        app: edge-scheduler
        component: scheduler-extender
        tier: control-plane
    spec:
      serviceAccountName: edge-scheduler
      containers:
      - name: edge-scheduler
        image: <your-registry>/edge-scheduler:latest  # 替换为实际镜像地址
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
          name: http
        envFrom:
        - configMapRef:
            name: edge-scheduler-config
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "1000m"
            memory: "1Gi"
        readinessProbe:
          httpGet:
            path: /api/v1/scheduler/health
            port: http
          initialDelaySeconds: 5
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /api/v1/scheduler/health
            port: http
          initialDelaySeconds: 15
          periodSeconds: 20
```

#### 使用 .env 文件方式

创建 `edge-scheduler-deployment-env.yaml` 文件：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: edge-scheduler
  namespace: edge-scheduler
  labels:
    app: edge-scheduler
spec:
  replicas: 1
  selector:
    matchLabels:
      app: edge-scheduler
  template:
    metadata:
      labels:
        app: edge-scheduler
    spec:
      serviceAccountName: edge-scheduler
      containers:
      - name: edge-scheduler
        image: <your-registry>/edge-scheduler:latest  # 替换为实际镜像地址
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics
        volumeMounts:
        - name: env-config
          mountPath: /app/.env
          subPath: .env
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "1000m"
            memory: "1Gi"
        readinessProbe:
          httpGet:
            path: /api/v1/scheduler/health
            port: http
          initialDelaySeconds: 5
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /api/v1/scheduler/health
            port: http
          initialDelaySeconds: 15
          periodSeconds: 20
      volumes:
      - name: env-config
        configMap:
          name: edge-scheduler-env-config
```

应用 Deployment：

```bash
# 使用键值对方式
kubectl apply -f edge-scheduler-deployment.yaml

# 或使用 .env 文件方式
kubectl apply -f edge-scheduler-deployment-env.yaml
```

### 6. 创建 Service

创建 `edge-scheduler-service.yaml` 文件：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: edge-scheduler
  namespace: edge-scheduler
spec:
  selector:
    app: edge-scheduler
  ports:
  - name: http
    port: 8000
    targetPort: http
  - name: metrics
    port: 9090
    targetPort: metrics
```

应用 Service：

```bash
kubectl apply -f edge-scheduler-service.yaml
```

## 环境变量配置说明

在新环境中部署时，需要特别关注以下环境变量：

### 必须修改的环境变量

1. **KUBE_NAMESPACE**：
   - 设置为部署 Edge Scheduler 的命名空间
   - 例如：`KUBE_NAMESPACE=edge-scheduler`

### 可选修改的环境变量

1. **LOG_LEVEL**：
   - 日志级别，可设置为 DEBUG、INFO、WARNING、ERROR
   - 生产环境建议使用 INFO 或 WARNING

2. **SCHEDULER_FILTER_TIMEOUT**：
   - 节点过滤超时时间，根据集群规模可能需要调整
   - 大型集群可能需要增加此值

3. **MAX_PODS_PER_NODE**：
   - 每个节点最大 Pod 数量，应与集群节点配置一致
   - 可通过 `kubectl describe node <node-name>` 查看节点的 `pods` 容量

4. **资源权重参数**：
   - `WEIGHT_CPU`、`WEIGHT_MEMORY`、`GA_WEIGHT_CPU` 等
   - 根据集群资源情况和调度偏好调整

## 验证部署

1. 检查 Pod 状态：

```bash
kubectl -n edge-scheduler get pods
```

2. 查看 Pod 日志：

```bash
kubectl -n edge-scheduler logs -f deployment/edge-scheduler
```

3. 测试 API 可用性：

```bash
# 获取 Service IP
kubectl -n edge-scheduler get svc

# 测试健康检查端点
curl http://<service-ip>:8000/api/v1/scheduler/health
```

## 故障排除

1. **Pod 无法启动**：
   - 检查镜像是否可访问
   - 查看 Pod 事件：`kubectl -n edge-scheduler describe pod <pod-name>`
   - 查看 Pod 日志：`kubectl -n edge-scheduler logs <pod-name>`

2. **无法连接到 Kubernetes API**：
   - 确认 ServiceAccount 和 RBAC 权限配置正确
   - 检查 `USE_SERVICE_ACCOUNT` 环境变量是否设置为 `true`

3. **无法获取监控数据**：
   - 确认 `MONITOR_API_URL` 配置正确
   - 检查网络连接是否正常
   - 验证 Prometheus/Thanos 服务是否可用

## 环境变量完整列表

| 环境变量 | 描述 | 默认值 | 是否必须修改 |
|---------|------|-------|------------|
| USE_SERVICE_ACCOUNT | 是否使用 ServiceAccount 进行身份验证 | true | 否 |
| KUBE_NAMESPACE | 默认命名空间 | edge-scheduler | 是 |
| APP_NAME | 应用名称 | edge-scheduler | 否 |
| APP_VERSION | 应用版本 | 1.0.0 | 否 |
| DEBUG | 是否启用调试模式 | false | 否 |
| LOG_LEVEL | 日志级别 | INFO | 否 |
| API_V1_PREFIX | API 前缀 | /v1 | 否 |
| HOST | 监听主机 | 0.0.0.0 | 否 |
| PORT | 监听端口 | 8000 | 否 |
| SCHEDULER_FILTER_TIMEOUT | 节点过滤超时时间（秒） | 5 | 可选 |
| MAX_PODS_PER_NODE | 每个节点最大 Pod 数量 | 110 | 可选 |
| ENABLE_GPU_SCHEDULING | 是否启用 GPU 调度 | false | 可选 |
| ENABLE_METRICS | 是否启用指标收集 | true | 否 |
| METRICS_PORT | 指标暴露端口 | 9090 | 否 |
| MONITOR_API_URL | 监控 API 地址 | http://thanos-querier:9091 | 是 |
| MONITOR_API_VERSION | 监控 API 版本 | v1 | 否 |
| MONITOR_REQUEST_TIMEOUT | 监控请求超时时间（秒） | 5 | 可选 |
| MONITOR_METRICS_WINDOW | 监控数据时间窗口（分钟） | 5 | 可选 |
| ENABLE_OPTIMIZER | 是否启用优化器 | true | 否 |
| WEIGHT_CPU | CPU 资源权重 | 0.6 | 可选 |
| WEIGHT_MEMORY | 内存资源权重 | 0.4 | 可选 |
| OPTIMIZER_ALGORITHM | 优化算法 | default | 否 |
| GA_POPULATION_SIZE | 遗传算法种群大小 | 50 | 可选 |
| GA_GENERATIONS | 遗传算法迭代代数 | 100 | 可选 |
| GA_MUTATION_RATE | 遗传算法变异率 | 0.1 | 可选 |
| GA_WEIGHT_CPU | 遗传算法 CPU 资源权重 | 0.5 | 可选 |
| GA_WEIGHT_MEMORY | 遗传算法内存资源权重 | 0.3 | 可选 |
| GA_WEIGHT_BALANCE | 遗传算法资源平衡权重 | 0.2 | 可选 |
| ENABLE_COMPUTING_POWER | 是否启用算力量化 | true | 否 |
| ENABLE_COMPUTING_POWER_FILTER | 是否启用算力量化筛选 | true | 否 |
| COMPUTING_ROUTER_HOST | 算力路由主机地址 | 127.0.0.1 | 否 |
| COMPUTING_ROUTER_PORT | 算力路由端口 | 8888 | 否 |
| UDP_SERVICE_PORT | 本地UDP服务端口 | 8889 | 否 |
| UDP_TIMEOUT | UDP请求超时时间（秒） | 5 | 否 |
| UDP_MAX_RETRIES | UDP请求最大重试次数 | 3 | 否 |
| UDP_RETRY_DELAY | 重试间隔（秒） | 1.0 | 否 |
| UDP_EXPONENTIAL_BACKOFF | 是否使用指数退避 | true | 否 |
| WEIGHT_CPU_POWER | CPU算力权重 | 0.5 | 否 |
| WEIGHT_GPU_POWER | GPU算力权重 | 0.3 | 否 |
| WEIGHT_FPGA_POWER | FPGA算力权重 | 0.2 | 否 |
| DEFAULT_CPU_POWER_RATIO | 默认CPU主频与核心数的乘积效率 | 0.8 | 否 |
| DEFAULT_GPU_TFLOPS | 默认GPU TFLOPS | 5.0 | 否 |
| DEFAULT_FPGA_TFLOPS | 默认FPGA TFLOPS | 2.0 | 否 |
| COMPUTING_POWER_FALLBACK | 降级策略 | default | 否 |
| TASK_PLANNING_ENABLED | 是否启用联合任务规划功能 | true | 否 |
| TASK_PLANNING_MAX_TASKS | 单个规划中最大任务数量 | 100 | 可选 |
| TASK_PLANNING_DEFAULT_NAMESPACE | 默认部署命名空间 | default | 可选 |
| TASK_PLANNING_DEPLOY_TIMEOUT | 任务部署超时时间（秒） | 300 | 可选 |
| TASK_PLANNING_RETRY_COUNT | 任务部署失败重试次数 | 3 | 可选 |
| TASK_PLANNING_RETRY_INTERVAL | 重试间隔时间（秒） | 5 | 可选 |

## 新环境对接指南

本章节提供将 Edge Scheduler 从一个 Kubernetes 环境迁移到另一个环境的详细步骤和最佳实践。

### 对接前准备

1. **环境评估清单**：
   - [ ] 确认新 K8s 集群版本（≥ 1.20.0）
   - [ ] 确认您拥有足够的集群权限（创建 namespace、deployment、service、RBAC 等）
   - [ ] 确认 kubectl 已正确配置可访问目标集群
   - [ ] 确认新环境中是否有可用的 Prometheus/Thanos 服务
   - [ ] 确认新环境的网络策略是否允许所需的连接

2. **资源需求评估**：
   - [ ] 确认新环境中节点资源是否满足 Edge Scheduler 的需求
   - [ ] 评估是否需要调整资源请求和限制
   - [ ] 确认新环境中每个节点的最大 Pod 数量限制

### 对接实施步骤

1. **环境信息收集**：
   ```bash
   # 获取集群版本信息
   kubectl version
   
   # 获取节点信息
   kubectl get nodes
   
   # 查看节点详情（包括资源和 Pod 容量）
   kubectl describe node <node-name>
   
   # 获取现有命名空间
   kubectl get namespaces
   
   # 检查 Prometheus/Thanos 服务（如果在特定命名空间）
   kubectl -n <monitoring-namespace> get svc
   ```

2. **命名空间规划**：
   - 决定在新环境中使用的命名空间名称
   - 如果需要使用非默认命名空间，确保在所有配置文件中一致更新

3. **环境变量调整**：
   
   创建一个环境变量配置文件模板，重点关注以下变量：
   
   ```yaml
   # 关键环境变量示例
   KUBE_NAMESPACE: "<新环境命名空间>"
   MONITOR_API_URL: "http://<新环境Prometheus服务地址>:<端口>"
   MAX_PODS_PER_NODE: "<根据新环境节点配置调整>"
   SCHEDULER_FILTER_TIMEOUT: "<根据集群规模调整>"
   ```

4. **镜像准备策略**：
   
   根据新环境的网络条件选择合适的镜像准备方式：
   
   - **有外部网络访问权限**：
     ```bash
     # 在源环境中标记镜像
     docker tag edge-scheduler:latest <新环境镜像仓库地址>/edge-scheduler:latest
     
     # 推送到新环境可访问的镜像仓库
     docker push <新环境镜像仓库地址>/edge-scheduler:latest
     ```
   
   - **无外部网络访问权限**：
     ```bash
     # 在源环境中导出镜像
     docker save -o edge-scheduler.tar edge-scheduler:latest
     
     # 将 tar 文件传输到新环境
     
     # 在新环境中导入镜像
     docker load -i edge-scheduler.tar
     ```

5. **配置文件准备**：
   
   准备以下配置文件，并根据新环境情况调整：
   
   - `edge-scheduler-configmap.yaml`（环境变量配置）
   - `edge-scheduler-rbac.yaml`（权限配置）
   - `edge-scheduler-deployment.yaml`（部署配置）
   - `edge-scheduler-service.yaml`（服务配置）

6. **分步部署与验证**：
   
   按顺序部署并在每步后验证：
   
   ```bash
   # 1. 创建命名空间
   kubectl create namespace <新环境命名空间>
   
   # 2. 应用 ConfigMap
   kubectl apply -f edge-scheduler-configmap.yaml
   
   # 3. 应用 RBAC 配置
   kubectl apply -f edge-scheduler-rbac.yaml
   
   # 4. 应用 Deployment
   kubectl apply -f edge-scheduler-deployment.yaml
   
   # 5. 应用 Service
   kubectl apply -f edge-scheduler-service.yaml
   ```

### 对接后验证

1. **基本功能验证**：
   ```bash
   # 检查 Pod 状态
   kubectl -n <新环境命名空间> get pods
   
   # 查看 Pod 日志
   kubectl -n <新环境命名空间> logs -f deployment/edge-scheduler
   
   # 检查服务是否正常创建
   kubectl -n <新环境命名空间> get svc
   ```

2. **健康检查**：
   ```bash
   # 获取服务 IP
   kubectl -n <新环境命名空间> get svc edge-scheduler -o jsonpath='{.spec.clusterIP}'
   
   # 测试健康检查端点
   curl http://<service-ip>:8000/api/v1/scheduler/health
   ```

3. **功能测试**：
   - 创建测试 Pod，验证调度功能
   - 检查监控指标是否正常收集
   - 验证与其他系统的集成

### 常见对接问题

1. **镜像拉取失败**：
   - 检查镜像仓库地址是否正确
   - 确认新环境是否需要镜像拉取凭证
   - 如需使用私有仓库，创建并配置 imagePullSecrets

2. **权限不足**：
   - 确认 RBAC 配置是否正确应用
   - 检查 ServiceAccount 是否正确创建和绑定
   - 验证 ClusterRole 权限是否足够

3. **监控集成问题**：
   - 确认 Prometheus/Thanos 服务地址是否正确
   - 检查网络策略是否允许 Edge Scheduler 访问监控服务
   - 验证监控服务 API 版本是否兼容

4. **资源限制问题**：
   - 如果 Pod 无法调度，检查节点资源是否足够
   - 调整资源请求和限制以适应新环境
   - 考虑使用节点亲和性规则确保部署在合适的节点上

### 对接检查清单

在完成部署后，使用以下检查清单确认所有关键功能：

- [ ] Pod 成功运行且状态为 Running
- [ ] 应用日志中没有错误信息
- [ ] 健康检查端点返回正常状态
- [ ] 应用可以正常访问 Kubernetes API
- [ ] 应用可以正常获取监控数据
- [ ] 资源使用在预期范围内
- [ ] 所有必要的环境变量已正确配置

完成以上检查后，Edge Scheduler 应已成功对接到新环境并正常运行。

## 算力度量服务配置

### 环境变量配置

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| COMPUTING_POWER_ENABLED | 是否启用算力度量功能 | true |
| UDP_SERVICE_HOST | UDP服务监听地址 | 0.0.0.0 |
| UDP_SERVICE_PORT | UDP服务监听端口 | 8001 |
| UDP_BUFFER_SIZE | UDP缓冲区大小（字节） | 4096 |
| UDP_REPORT_TIMEOUT | 等待算力信息上报超时时间（秒） | 5.0 |
| COMPUTING_POWER_CLEANUP_INTERVAL | 清理过期算力信息的间隔（秒） | 300 |
| COMPUTING_POWER_EXPIRY_TIME | 算力信息过期时间（秒） | 600 |

### ConfigMap配置示例

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: edge-scheduler-config
  namespace: edge-system
data:
  COMPUTING_POWER_ENABLED: "true"
  UDP_SERVICE_HOST: "0.0.0.0"
  UDP_SERVICE_PORT: "8001"
  UDP_BUFFER_SIZE: "4096"
  UDP_REPORT_TIMEOUT: "5.0"
  COMPUTING_POWER_CLEANUP_INTERVAL: "300"
  COMPUTING_POWER_EXPIRY_TIME: "600"
```

### 验证步骤

1. 检查UDP服务是否正常启动：
```bash
netstat -nulp | grep 8001
```

2. 使用测试工具发送算力信息：
```python
import socket
import struct
import time

def send_test_report():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # 构造测试数据
    report_id = 1
    host_number = 1
    host_name = "test-node-1"
    total_memory = 100
    available_memory = 80
    computing_force_type = 1  # CPU
    cpu_count = 1
    computing_power = 2.5
    load = 60
    
    # 打包数据
    data = struct.pack('!I', report_id)  # 报告ID (4字节)
    data += struct.pack('!B', host_number)  # 节点数量 (1字节)
    data += host_name.encode().ljust(64, b'\x00')  # 节点名称 (64字节)
    data += struct.pack('!BB', total_memory, available_memory)  # 内存信息 (2字节)
    data += struct.pack('!B', computing_force_type)  # 算力类型 (1字节)
    data += struct.pack('!B', cpu_count)  # CPU数量 (1字节)
    data += struct.pack('!f', computing_power)[:3]  # CPU算力 (3字节)
    data += struct.pack('!B', load)  # CPU负载 (1字节)
    
    # 发送数据
    sock.sendto(data, ('localhost', 8001))
    sock.close()

send_test_report()
```

3. 检查服务日志：
```bash
kubectl logs -n edge-system -l app=edge-scheduler | grep "算力信息上报"
```

### 监控指标

服务提供以下Prometheus监控指标：

- `computing_power_reports_total`: 接收到的算力信息上报总数
- `computing_power_parse_errors_total`: 解析算力信息失败次数
- `computing_power_latest_report_timestamp`: 最新算力信息上报时间戳
- `computing_power_nodes_total`: 当前活跃的节点总数

### 故障排查

1. UDP服务无法启动
- 检查端口是否被占用
- 确认服务账号是否有足够权限
- 查看服务日志中的详细错误信息

2. 无法接收算力信息
- 确认发送方的数据格式是否正确
- 检查网络连接是否正常
- 验证防火墙规则是否允许UDP流量

3. 算力信息解析失败
- 检查数据包格式是否符合协议规范
- 确认字节序是否正确
- 查看服务日志中的解析错误详情

## 资源状态服务配置

### 环境变量配置

资源状态服务相关的环境变量配置：

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| RESOURCE_STATUS_ENABLED | 是否启用资源状态服务 | true |
| RESOURCE_STATUS_UPDATE_INTERVAL | 资源状态更新间隔(秒) | 5 |
| RESOURCE_STATUS_CLEANUP_INTERVAL | 断开连接清理间隔(秒) | 60 |
| RESOURCE_STATUS_MAX_CONNECTIONS | 最大WebSocket连接数 | 100 |
| RESOURCE_STATUS_BROADCAST_TIMEOUT | 广播超时时间(秒) | 5 |

### ConfigMap配置示例

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: edge-scheduler-config
  namespace: edge-system
data:
  RESOURCE_STATUS_ENABLED: "true"
  RESOURCE_STATUS_UPDATE_INTERVAL: "5"
  RESOURCE_STATUS_CLEANUP_INTERVAL: "60"
  RESOURCE_STATUS_MAX_CONNECTIONS: "100"
  RESOURCE_STATUS_BROADCAST_TIMEOUT: "5"
```

### ServiceAccount权限配置

资源状态服务需要以下权限来监控和管理资源状态：

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: resource-status-role
rules:
- apiGroups: [""]
  resources: ["nodes", "pods"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["metrics.k8s.io"]
  resources: ["nodes", "pods"]
  verbs: ["get", "list"]
```

### 验证步骤

1. 检查服务健康状态：
```bash
curl http://<service-ip>:8000/api/v1/resource-status/health
```

2. 测试WebSocket连接：
```python
import websockets
import asyncio
import json

async def test_connection():
    uri = f"ws://<service-ip>:8000/api/v1/resource-status/ws"
    async with websockets.connect(uri) as websocket:
        query = {
            "request_id": "test-001",
            "query": "all"
        }
        await websocket.send(json.dumps(query))
        response = await websocket.recv()
        print(f"连接测试成功: {response}")

asyncio.run(test_connection())
```

3. 检查日志输出：
```bash
kubectl logs -n edge-system -l app=edge-scheduler -c edge-scheduler | grep "resource-status"
```

### 监控指标

资源状态服务提供以下Prometheus监控指标：

- `resource_status_connections_total`: 当前活动WebSocket连接数
- `resource_status_messages_total`: 消息处理总数
- `resource_status_errors_total`: 错误发生总数
- `resource_status_broadcast_duration_seconds`: 广播消息耗时分布

### 故障排除

1. WebSocket连接失败：
   - 检查服务是否正常运行
   - 验证网络连接和防火墙配置
   - 查看服务日志中的错误信息

2. 资源状态更新失败：
   - 确认请求格式是否正确
   - 检查节点ID是否存在
   - 验证数值范围是否合法

3. 广播消息延迟：
   - 检查网络延迟
   - 调整广播超时时间
   - 优化连接数量 