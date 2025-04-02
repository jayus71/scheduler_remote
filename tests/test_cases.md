# 星载微服务协同系统测试用例

## 1. 过滤器接口测试用例 (Filter API)

### 测试用例 1.1: 基本过滤功能测试

**测试目标**: 验证过滤器能够根据Pod资源需求正确过滤节点

**测试步骤**:
1. 准备测试Pod (nginx-pod)，资源需求为 CPU: 500m, Memory: 256Mi
2. 准备测试节点列表，包含3个worker节点，资源情况各不相同
3. 发送过滤请求到 `/api/v1/filter` 接口

**请求数据**:
```json
{
  "pod": {
    "metadata": {
      "name": "nginx-pod",
      "namespace": "test-scheduler",
      "uid": "test-uid-nginx"
    },
    "spec": {
      "containers": [
        {
          "name": "nginx",
          "image": "nginx:latest",
          "resources": {
            "requests": {
              "cpu": "500m",
              "memory": "256Mi"
            },
            "limits": {
              "cpu": "1000m",
              "memory": "512Mi"
            }
          }
        }
      ],
      "nodeName": null
    }
  },
  "nodes": [
    {
      "name": "kind-worker",
      "labels": {
        "kubernetes.io/hostname": "kind-worker"
      },
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi"
      },
      "allocated": {
        "cpu": "1",
        "memory": "2Gi"
      }
    },
    {
      "name": "kind-worker2",
      "labels": {
        "kubernetes.io/hostname": "kind-worker2"
      },
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi"
      },
      "allocated": {
        "cpu": "3.5",
        "memory": "7Gi"
      }
    },
    {
      "name": "kind-worker3",
      "labels": {
        "kubernetes.io/hostname": "kind-worker3"
      },
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi"
      },
      "allocated": {
        "cpu": "2",
        "memory": "4Gi"
      }
    }
  ]
}
```

**预期结果**:
- 返回状态码 200
- 返回符合条件的节点列表，应包含 kind-worker 和 kind-worker3
- kind-worker2 应在 failed_nodes 列表中，因为资源不足

### 测试用例 1.2: 资源密集型Pod过滤测试

**测试目标**: 验证过滤器能够正确处理资源需求较高的Pod

**测试步骤**:
1. 准备测试Pod (resource-heavy-pod)，资源需求为 CPU: 3000m, Memory: 6Gi
2. 准备测试节点列表，包含3个worker节点
3. 发送过滤请求到 `/api/v1/filter` 接口

**请求数据**:
```json
{
  "pod": {
    "metadata": {
      "name": "resource-heavy-pod",
      "namespace": "test-scheduler",
      "uid": "test-uid-heavy"
    },
    "spec": {
      "containers": [
        {
          "name": "resource-heavy",
          "resources": {
            "requests": {
              "cpu": "3000m",
              "memory": "6Gi"
            },
            "limits": {
              "cpu": "4000m",
              "memory": "8Gi"
            }
          }
        }
      ]
    }
  },
  "nodes": [
    {
      "name": "kind-worker",
      "labels": {
        "kubernetes.io/hostname": "kind-worker"
      },
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi"
      },
      "allocated": {
        "cpu": "1",
        "memory": "2Gi"
      }
    },
    {
      "name": "kind-worker2",
      "labels": {
        "kubernetes.io/hostname": "kind-worker2"
      },
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi"
      },
      "allocated": {
        "cpu": "3.5",
        "memory": "7Gi"
      }
    },
    {
      "name": "kind-worker3",
      "labels": {
        "kubernetes.io/hostname": "kind-worker3"
      },
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi"
      },
      "allocated": {
        "cpu": "2",
        "memory": "4Gi"
      }
    }
  ]
}
```

**预期结果**:
- 返回状态码 200
- 返回符合条件的节点列表，应只包含 kind-worker
- kind-worker2 和 kind-worker3 应在 failed_nodes 列表中，因为资源不足

### 测试用例 1.3: 空节点列表测试

**测试目标**: 验证过滤器能够正确处理空节点列表

**测试步骤**:
1. 准备测试Pod (nginx-pod)
2. 发送包含空节点列表的过滤请求

**请求数据**:
```json
{
  "pod": {
    "metadata": {
      "name": "nginx-pod",
      "namespace": "test-scheduler",
      "uid": "test-uid-nginx"
    },
    "spec": {
      "containers": [
        {
          "name": "nginx",
          "resources": {
            "requests": {
              "cpu": "500m",
              "memory": "256Mi"
            }
          }
        }
      ]
    }
  },
  "nodes": []
}
```

**预期结果**:
- 返回状态码 200
- 返回空的符合条件节点列表
- 返回空的失败节点列表

### 测试用例 1.4: 节点选择器测试

**测试目标**: 验证过滤器能够正确处理带有节点选择器的Pod

**测试步骤**:
1. 准备带有节点选择器的测试Pod (nginx-pod)
2. 准备测试节点列表，包含带有不同标签的节点
3. 发送过滤请求到 `/api/v1/filter` 接口

**请求数据**:
```json
{
  "pod": {
    "metadata": {
      "name": "nginx-pod",
      "namespace": "test-scheduler",
      "uid": "test-uid-nginx"
    },
    "spec": {
      "containers": [
        {
          "name": "nginx",
          "resources": {
            "requests": {
              "cpu": "500m",
              "memory": "256Mi"
            }
          }
        }
      ],
      "node_selector": {
        "environment": "production"
      }
    }
  },
  "nodes": [
    {
      "name": "kind-worker",
      "labels": {
        "kubernetes.io/hostname": "kind-worker",
        "environment": "production"
      },
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi"
      },
      "allocated": {
        "cpu": "1",
        "memory": "2Gi"
      }
    },
    {
      "name": "kind-worker2",
      "labels": {
        "kubernetes.io/hostname": "kind-worker2",
        "environment": "staging"
      },
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi"
      },
      "allocated": {
        "cpu": "1",
        "memory": "2Gi"
      }
    },
    {
      "name": "kind-worker3",
      "labels": {
        "kubernetes.io/hostname": "kind-worker3",
        "environment": "production"
      },
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi"
      },
      "allocated": {
        "cpu": "2",
        "memory": "4Gi"
      }
    }
  ]
}
```

**预期结果**:
- 返回状态码 200
- 返回符合条件的节点列表，应包含 kind-worker 和 kind-worker3
- kind-worker2 应在 failed_nodes 列表中，因为标签不匹配

### 测试用例 1.5: 算力需求测试

**测试目标**: 验证过滤器能够正确处理带有算力需求的Pod

**测试步骤**:
1. 准备需要GPU资源的测试Pod (gpu-pod)
2. 准备测试节点列表，包含带有不同算力资源的节点
3. 发送过滤请求到 `/api/v1/filter` 接口

**请求数据**:
```json
{
  "pod": {
    "metadata": {
      "name": "gpu-pod",
      "namespace": "test-scheduler",
      "uid": "test-uid-gpu"
    },
    "spec": {
      "containers": [
        {
          "name": "gpu-container",
          "image": "nvidia/cuda:11.0-base",
          "resources": {
            "requests": {
              "cpu": "1000m",
              "memory": "2Gi",
              "nvidia.com/gpu": "1"
            },
            "limits": {
              "cpu": "2000m",
              "memory": "4Gi",
              "nvidia.com/gpu": "1"
            }
          }
        }
      ],
      "nodeName": null
    }
  },
  "nodes": [
    {
      "name": "kind-worker",
      "labels": {
        "kubernetes.io/hostname": "kind-worker"
      },
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi"
      },
      "allocated": {
        "cpu": "1",
        "memory": "2Gi"
      }
    },
    {
      "name": "kind-worker2",
      "labels": {
        "kubernetes.io/hostname": "kind-worker2"
      },
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110",
        "nvidia.com/gpu": "2"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi",
        "nvidia.com/gpu": "2"
      },
      "allocated": {
        "cpu": "1",
        "memory": "2Gi",
        "nvidia.com/gpu": "1"
      }
    },
    {
      "name": "kind-worker3",
      "labels": {
        "kubernetes.io/hostname": "kind-worker3"
      },
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi"
      },
      "allocated": {
        "cpu": "2",
        "memory": "4Gi"
      }
    }
  ]
}
```

**预期结果**:
- 返回状态码 200
- 返回符合条件的节点列表，应只包含 kind-worker2
- kind-worker 和 kind-worker3 应在 failed_nodes 列表中，因为缺少GPU资源

## 2. 节点绑定接口测试用例 (Bind API)

### 测试用例 2.1: 基本绑定功能测试

**测试目标**: 验证绑定接口能够正确将Pod绑定到指定节点

**测试步骤**:
1. 准备测试Pod (nginx-pod)
2. 发送绑定请求到 `/api/v1/bind` 接口

**请求数据**:
```json
{
  "pod_name": "nginx-pod",
  "pod_namespace": "test-scheduler",
  "pod_uid": "test-uid-nginx",
  "node": "kind-worker"
}
```

**预期结果**:
- 返回状态码 200
- 返回空的错误信息，表示绑定成功
- Pod成功绑定到指定节点

### 测试用例 2.2: 与调度方案集成测试

**测试目标**: 验证绑定接口能够正确使用调度方案中的选定节点

**测试步骤**:
1. 准备测试Pod (redis-pod)
2. 通过优先级接口生成调度方案，选定节点为 kind-worker3
3. 发送绑定请求到 `/api/v1/bind` 接口，指定节点为 kind-worker

**请求数据**:
```json
{
  "pod_name": "redis-pod",
  "pod_namespace": "test-scheduler",
  "pod_uid": "test-uid-redis",
  "node": "kind-worker"
}
```

**预期结果**:
- 返回状态码 200
- 返回空的错误信息，表示绑定成功
- Pod应被绑定到调度方案中的选定节点 kind-worker3，而不是请求中的 kind-worker
- 调度方案应被清除

### 测试用例 2.3: 无效节点绑定测试

**测试目标**: 验证绑定接口能够正确处理无效节点

**测试步骤**:
1. 准备测试Pod (nginx-pod)
2. 发送绑定请求到 `/api/v1/bind` 接口，指定不存在的节点

**请求数据**:
```json
{
  "pod_name": "nginx-pod",
  "pod_namespace": "test-scheduler",
  "pod_uid": "test-uid-nginx",
  "node": "non-existent-node"
}
```

**预期结果**:
- 返回状态码 200
- 返回包含错误信息的响应，表示绑定失败
- Pod不应被绑定到任何节点

## 3. 网络接口测试用例 (Network API)

### 测试用例 3.1: 获取网络信息测试

**测试目标**: 验证网络接口能够正确返回网络信息

**测试步骤**:
1. 发送GET请求到 `/api/v1/network/info` 接口

**预期结果**:
- 返回状态码 200
- 返回包含所有网络信息的响应

### 测试用例 3.2: 获取特定节点间网络信息测试

**测试目标**: 验证网络接口能够正确返回特定节点间的网络信息

**测试步骤**:
1. 发送GET请求到 `/api/v1/network/info?source_node=kind-worker&target_node=kind-worker2` 接口

**预期结果**:
- 返回状态码 200
- 返回包含指定节点间网络信息的响应，包括延迟和带宽

### 测试用例 3.3: 网络信息报告测试

**测试目标**: 验证网络接口能够正确处理网络信息报告

**测试步骤**:
1. 准备网络信息报告
2. 发送POST请求到 `/api/v1/network/test` 接口

**请求数据**:
```json
{
  "report_id": "test-report-001",
  "mirror_repository_host_name": "kind-worker",
  "destinations": [
    {
      "node_name": "kind-worker2",
      "latency": 10,
      "bandwidth": 100
    },
    {
      "node_name": "kind-worker3",
      "latency": 15,
      "bandwidth": 90
    }
  ]
}
```

**预期结果**:
- 返回状态码 200
- 返回成功处理报告的响应
- 网络信息应被正确存储

## 4. 优先级接口测试用例 (Priority API)

### 测试用例 4.1: 基本优先级计算测试

**测试目标**: 验证优先级接口能够正确计算节点优先级

**测试步骤**:
1. 准备测试Pod (nginx-pod)
2. 准备测试节点列表
3. 发送优先级计算请求到 `/api/v1/scheduler/priority` 接口

**请求数据**:
```json
{
  "pod": {
    "metadata": {
      "name": "nginx-pod",
      "namespace": "test-scheduler",
      "uid": "test-uid-nginx"
    },
    "spec": {
      "containers": [
        {
          "name": "nginx",
          "image": "nginx:latest",
          "resources": {
            "requests": {
              "cpu": "500m",
              "memory": "256Mi"
            },
            "limits": {
              "cpu": "1000m",
              "memory": "512Mi"
            }
          }
        }
      ],
      "nodeName": null
    }
  },
  "nodes": [
    {
      "name": "kind-worker",
      "labels": {},
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi"
      },
      "allocated": {
        "cpu": "1",
        "memory": "2Gi"
      }
    },
    {
      "name": "kind-worker2",
      "labels": {},
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi"
      },
      "allocated": {
        "cpu": "2",
        "memory": "4Gi"
      }
    },
    {
      "name": "kind-worker3",
      "labels": {},
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi"
      },
      "allocated": {
        "cpu": "3",
        "memory": "6Gi"
      }
    }
  ]
}
```

**预期结果**:
- 返回状态码 200
- 返回包含节点优先级列表的响应
- 节点优先级应根据资源使用情况排序，资源空闲较多的节点得分较高

### 测试用例 4.2: 调度方案查询测试

**测试目标**: 验证能够正确查询生成的调度方案

**测试步骤**:
1. 通过优先级接口为Pod (nginx-pod) 生成调度方案
2. 发送GET请求到 `/api/v1/scheduler/plan/nginx-pod` 接口

**预期结果**:
- 返回状态码 200
- 返回包含Pod调度方案的响应，包括节点得分和选定节点

### 测试用例 4.3: 节点得分查询测试

**测试目标**: 验证能够正确查询节点得分

**测试步骤**:
1. 通过优先级接口为Pod (nginx-pod) 生成调度方案
2. 发送GET请求到 `/api/v1/scheduler/scores/nginx-pod` 接口

**预期结果**:
- 返回状态码 200
- 返回包含节点得分列表的响应

### 测试用例 4.4: 带有算力信息的优先级计算测试

**测试目标**: 验证优先级接口能够正确处理带有算力信息的请求

**测试步骤**:
1. 准备测试Pod (gpu-pod)
2. 准备测试节点列表和算力信息
3. 发送优先级计算请求到 `/api/v1/scheduler/priority` 接口

**请求数据**:
```json
{
  "pod": {
    "metadata": {
      "name": "gpu-pod",
      "namespace": "test-scheduler",
      "uid": "test-uid-gpu"
    },
    "spec": {
      "containers": [
        {
          "name": "gpu-container",
          "image": "nvidia/cuda:11.0-base",
          "resources": {
            "requests": {
              "cpu": "1000m",
              "memory": "2Gi",
              "nvidia.com/gpu": "1"
            },
            "limits": {
              "cpu": "2000m",
              "memory": "4Gi",
              "nvidia.com/gpu": "1"
            }
          }
        }
      ],
      "nodeName": null
    }
  },
  "nodes": [
    {
      "name": "kind-worker",
      "labels": {},
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi"
      },
      "allocated": {
        "cpu": "1",
        "memory": "2Gi"
      }
    },
    {
      "name": "kind-worker2",
      "labels": {},
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110",
        "nvidia.com/gpu": "2"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi",
        "nvidia.com/gpu": "2"
      },
      "allocated": {
        "cpu": "1",
        "memory": "2Gi",
        "nvidia.com/gpu": "1"
      }
    },
    {
      "name": "kind-worker3",
      "labels": {},
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi"
      },
      "allocated": {
        "cpu": "2",
        "memory": "4Gi"
      }
    }
  ],
  "computing_power_info": {
    "kind-worker": {
      "cpu_load": 0.3,
      "gpu_load": 0.0,
      "fpga_load": 0.0
    },
    "kind-worker2": {
      "cpu_load": 0.4,
      "gpu_load": 0.5,
      "fpga_load": 0.0
    },
    "kind-worker3": {
      "cpu_load": 0.6,
      "gpu_load": 0.0,
      "fpga_load": 0.0
    }
  }
}
```

**预期结果**:
- 返回状态码 200
- 返回包含节点优先级列表的响应
- 节点优先级应考虑算力信息，具有GPU资源的节点 kind-worker2 应该得分较高

### 测试用例 4.5: 带有网络信息的优先级计算测试

**测试目标**: 验证优先级接口能够正确处理带有网络信息的请求

**测试步骤**:
1. 准备测试Pod (nginx-pod)
2. 准备测试节点列表和网络信息
3. 发送优先级计算请求到 `/api/v1/scheduler/priority` 接口

**请求数据**:
```json
{
  "pod": {
    "metadata": {
      "name": "nginx-pod",
      "namespace": "test-scheduler",
      "uid": "test-uid-nginx"
    },
    "spec": {
      "containers": [
        {
          "name": "nginx",
          "image": "nginx:latest",
          "resources": {
            "requests": {
              "cpu": "500m",
              "memory": "256Mi"
            },
            "limits": {
              "cpu": "1000m",
              "memory": "512Mi"
            }
          }
        }
      ]
    }
  },
  "nodes": [
    {
      "name": "kind-worker",
      "labels": {},
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi"
      },
      "allocated": {
        "cpu": "1",
        "memory": "2Gi"
      }
    },
    {
      "name": "kind-worker2",
      "labels": {},
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi"
      },
      "allocated": {
        "cpu": "2",
        "memory": "4Gi"
      }
    },
    {
      "name": "kind-worker3",
      "labels": {},
      "capacity": {
        "cpu": "4",
        "memory": "8Gi",
        "pods": "110"
      },
      "allocatable": {
        "cpu": "4",
        "memory": "8Gi"
      },
      "allocated": {
        "cpu": "3",
        "memory": "6Gi"
      }
    }
  ],
  "network_info": {
    "kind-worker": {
      "latency": 5,
      "bandwidth": 1000
    },
    "kind-worker2": {
      "latency": 10,
      "bandwidth": 800
    },
    "kind-worker3": {
      "latency": 15,
      "bandwidth": 600
    }
  }
}
```

**预期结果**:
- 返回状态码 200
- 返回包含节点优先级列表的响应
- 节点优先级应考虑网络信息，网络延迟低、带宽高的节点 kind-worker 应该得分较高

## 5. 路由信息接口测试用例 (Route Info API)

### 测试用例 5.1: WebSocket连接测试

**测试目标**: 验证路由信息WebSocket接口能够正确建立连接

**测试步骤**:
1. 建立WebSocket连接到 `/api/v1/route-info/ws` 端点

**预期结果**:
- 连接成功建立
- 接收到欢迎消息

### 测试用例 5.2: 路由信息查询测试

**测试目标**: 验证路由信息接口能够正确处理路由信息查询请求

**测试步骤**:
1. 建立WebSocket连接
2. 发送路由信息查询请求

**请求数据**:
```json
{
  "request_id": "test-request-001",
  "source_nodes": ["kind-worker"],
  "target_nodes": ["kind-worker2", "kind-worker3"]
}
```

**预期结果**:
- 接收到包含路由信息的响应
- 响应应包含源节点到目标节点的路由信息，包括连接状态、延迟、带宽和路由路径

### 测试用例 5.3: 全局路由信息查询测试

**测试目标**: 验证路由信息接口能够正确处理全局路由信息查询请求

**测试步骤**:
1. 建立WebSocket连接
2. 发送不指定源节点和目标节点的路由信息查询请求

**请求数据**:
```json
{
  "request_id": "test-request-002"
}
```

**预期结果**:
- 接收到包含所有节点间路由信息的响应
- 响应应包含所有可用节点间的路由信息

## 6. 资源状态接口测试用例 (Resource Status API)

### 测试用例 6.1: WebSocket连接测试

**测试目标**: 验证资源状态WebSocket接口能够正确建立连接

**测试步骤**:
1. 建立WebSocket连接到 `/api/v1/resource-status/ws` 端点

**预期结果**:
- 连接成功建立

### 测试用例 6.2: 资源状态更新测试

**测试目标**: 验证资源状态接口能够正确处理资源状态更新请求

**测试步骤**:
1. 建立WebSocket连接
2. 发送资源状态更新请求

**请求数据**:
```json
{
  "request_id": "test-request-001",
  "node_id": "kind-worker",
  "status_id": "status-001",
  "cpu_status": "available",
  "cpu_usage": 45.5,
  "gpu_status": "in_use",
  "gpu_usage": 78.3,
  "memory_status": "available",
  "memory_usage": 60.0,
  "disk_status": "available",
  "disk_usage": 55.0,
  "network_status": "available",
  "network_usage": 30.0,
  "overall_load": 58.2
}
```

**预期结果**:
- 接收到确认资源状态更新成功的响应
- 资源状态应被正确存储

### 测试用例 6.3: 资源状态查询测试

**测试目标**: 验证资源状态接口能够正确处理资源状态查询请求

**测试步骤**:
1. 建立WebSocket连接
2. 更新多个节点的资源状态
3. 发送资源状态查询请求

**请求数据**:
```json
{
  "request_id": "test-request-002",
  "query": "all"
}
```

**预期结果**:
- 接收到包含所有节点资源状态的响应
- 响应应包含之前更新的所有节点的资源状态信息

### 测试用例 6.4: 特定节点资源状态查询测试

**测试目标**: 验证资源状态接口能够正确处理特定节点的资源状态查询请求

**测试步骤**:
1. 建立WebSocket连接
2. 更新多个节点的资源状态
3. 发送特定节点的资源状态查询请求

**请求数据**:
```json
{
  "request_id": "test-request-003",
  "query": "kind-worker"
}
```

**预期结果**:
- 接收到包含特定节点资源状态的响应
- 响应应只包含指定节点的资源状态信息

### 测试用例 6.5: 健康检查测试

**测试目标**: 验证资源状态接口的健康检查功能

**测试步骤**:
1. 发送GET请求到 `/api/v1/resource-status/health` 接口

**预期结果**:
- 返回状态码 200
- 返回包含服务健康状态和活动连接数的响应