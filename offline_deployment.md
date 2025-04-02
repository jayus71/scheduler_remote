# 离线部署指南

本文档提供了在离线环境中部署星载微服务协同系统的详细步骤。

## 准备工作

### 1. 导出镜像

在有网络连接的环境中，执行以下步骤导出所需的Docker镜像：

```bash
# 构建应用镜像
docker build -t edge-scheduler:latest .

# 导出镜像为tar文件
docker save -o edge-scheduler.tar edge-scheduler:latest
```

### 2. 准备依赖文件

确保以下文件已准备好：

- 应用代码和配置文件
- Docker镜像tar文件
- .env配置文件

## 离线部署步骤

### 1. 导入Docker镜像

在离线环境中，执行以下命令导入Docker镜像：

```bash
# 导入应用镜像
docker load -i edge-scheduler.tar
```

### 2. 配置环境

1. 创建必要的目录结构：

```bash
mkdir -p config logs
```

2. 复制配置文件到config目录：

```bash
cp -r /path/to/config/* ./config/
```

3. 创建并配置.env文件：

```bash
# 创建.env文件
cat > .env << EOF
# 基本配置
APP_NAME=edge-scheduler
LOG_LEVEL=INFO

# Kubernetes配置
USE_SERVICE_ACCOUNT=false
# 在离线环境中启用K8s模拟模式
MOCK_K8S=true

# UDP服务配置
UDP_SERVICE_HOST=0.0.0.0
UDP_SERVICE_PORT=9001
COMPUTING_POWER_PORT=9002
EOF

# 检查.env文件
cat .env

# 如需修改，可以使用编辑器
# vim .env
```

### 3. 启动服务

使用docker run启动服务，从.env文件加载环境变量：

```bash
# 启动主服务
docker run -d --name edge-scheduler \
  -p 8000:8000 \
  -p 9001:9001/udp \
  -p 9002:9002/udp \
  --env-file .env \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/logs:/app/logs \
  edge-scheduler:latest
```

### 4. 验证部署

检查服务是否正常运行：

```bash
# 查看容器状态
docker ps

# 查看服务日志
docker logs -f edge-scheduler

# 检查健康状态
curl http://localhost:8000/health
```

## 离线环境特殊配置

在离线环境中，可以通过.env文件配置以下特殊设置：

1. 启用K8s模拟模式，避免需要真实的Kubernetes集群：

```
MOCK_K8S=true
```

2. 如果不需要连接到Kubernetes集群，可以在.env文件中设置：

```
USE_SERVICE_ACCOUNT=false
```

3. 对于网络和算力信息服务，确保在.env文件中正确配置UDP端口：

```
UDP_SERVICE_HOST=0.0.0.0
UDP_SERVICE_PORT=9001
COMPUTING_POWER_PORT=9002
```

## 故障排除

### 常见问题

1. **容器无法启动**
   - 检查日志：`docker logs edge-scheduler`
   - 确认配置文件和目录权限正确
   - 检查.env文件配置是否正确

2. **健康检查失败**
   - 检查应用日志是否有错误
   - 确认端口映射是否正确：`docker ps`

3. **Kubernetes连接错误**
   - 确保.env文件中设置了`MOCK_K8S=true`
   - 如果需要连接真实集群，检查kubeconfig.yaml文件是否正确配置

4. **UDP服务无法接收数据**
   - 确认UDP端口已正确映射：`docker ps`
   - 检查防火墙设置是否允许UDP流量
   - 验证.env文件中的UDP配置是否正确

### 日志收集

如需收集日志进行问题排查：

```bash
# 收集容器日志
docker logs edge-scheduler > edge-scheduler-logs.txt

# 收集应用日志
tar -czf app-logs.tar.gz logs/
```

## 更新部署

如需更新离线部署：

1. 在有网络的环境中构建新版本镜像并导出
2. 在离线环境中导入新镜像
3. 更新服务：

```bash
# 停止并删除当前容器
docker stop edge-scheduler
docker rm edge-scheduler

# 启动更新后的服务
docker run -d --name edge-scheduler \
  -p 8000:8000 \
  -p 9001:9001/udp \
  -p 9002:9002/udp \
  --env-file .env \
  -v $(pwd)/config:/app/config:ro \
  -v $(pwd)/logs:/app/logs \
  edge-scheduler:latest
```

## 容器管理命令

```bash
# 查看容器日志
docker logs -f edge-scheduler

# 停止容器
docker stop edge-scheduler

# 启动已停止的容器
docker start edge-scheduler

# 重启容器
docker restart edge-scheduler

# 删除容器
docker rm edge-scheduler
``` 