"""强化学习代理模块"""
import os
import time
import random
import numpy as np
import math
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from loguru import logger

from app.core.config import settings
from app.schemas.priority import RLState, RLAction, RLExperience, RLModelInfo


class DQNetwork(nn.Module):
    """深度Q网络模型"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """初始化DQN网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(DQNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入状态张量
            
        Returns:
            动作Q值张量
        """
        return self.network(x)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        """初始化经验回放缓冲区
        
        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
        
    def add(self, experience: RLExperience):
        """添加经验
        
        Args:
            experience: 经验对象
        """
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[RLExperience]:
        """采样经验批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            经验批次列表
        """
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def __len__(self) -> int:
        """获取缓冲区长度"""
        return len(self.buffer)


class RLAgent:
    """强化学习代理"""
    
    def __init__(self):
        """初始化强化学习代理"""
        self.algorithm = settings.RL_ALGORITHM
        self.learning_rate = settings.RL_LEARNING_RATE
        self.gamma = settings.RL_GAMMA  # 折扣因子
        self.epsilon = settings.RL_EPSILON  # 探索率
        self.epsilon_min = settings.RL_EPSILON_MIN  # 最小探索率
        self.epsilon_decay = settings.RL_EPSILON_DECAY  # 探索率衰减系数
        self.batch_size = settings.RL_BATCH_SIZE  # 批大小
        self.model_path = settings.RL_MODEL_PATH  # 模型保存路径
        self.target_update_freq = settings.RL_TARGET_UPDATE_FREQ  # 目标网络更新频率
        self.training_steps = 0  # 训练步数
        self.model_save_interval = settings.RL_MODEL_SAVE_INTERVAL  # 模型保存间隔
        
        # 创建模型保存目录
        os.makedirs(self.model_path, exist_ok=True)
        
        # 经验回放缓冲区
        self.memory = ReplayBuffer(settings.RL_MEMORY_SIZE)
        
        # 延迟初始化网络
        self.state_dim = None
        self.action_dim = None
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        self.initialized = False
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"RLAgent 初始化完成，使用设备: {self.device}")
        
        # 尝试加载现有模型
        self.load_model()
        
    def _initialize_networks(self, state_dim: int, action_dim: int):
        """初始化网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
        """
        if self.initialized:
            return
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 创建Q网络和目标网络
        self.q_network = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNetwork(state_dim, action_dim).to(self.device)
        
        # 初始化目标网络权重为当前网络权重
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 设置优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        self.initialized = True
        logger.info(f"DQN网络初始化完成，状态维度: {state_dim}, 动作维度: {action_dim}")
        
    def preprocess_state(self, state: RLState) -> torch.Tensor:
        """预处理状态为张量
        
        Args:
            state: 状态对象
            
        Returns:
            处理后的状态张量
        """
        # 提取节点特征
        node_features = state.node_features
        nodes = list(node_features.keys())
        
        # 计算每个节点特征的平均值作为状态表示
        features = []
        for node in nodes:
            features.extend(node_features[node])
        
        # 转换为张量
        state_tensor = torch.FloatTensor(features).to(self.device)
        
        return state_tensor
    
    def select_action(self, state: RLState, 
                     available_nodes: List[str], 
                     training: bool = True) -> Dict[str, float]:
        """选择动作
        
        Args:
            state: 状态
            available_nodes: 可用节点列表
            training: 是否在训练模式
            
        Returns:
            动作（节点权重字典）
        """
        # 状态预处理
        state_tensor = self.preprocess_state(state)
        
        # 如果网络还未初始化，初始化网络
        if not self.initialized:
            self._initialize_networks(len(state_tensor), len(available_nodes))
        
        # 探索-利用平衡（ε-greedy策略）
        if training and random.random() < self.epsilon:
            # 探索：随机选择
            action_weights = [random.random() for _ in range(len(available_nodes))]
            # 归一化
            total = sum(action_weights)
            if total > 0:
                action_weights = [w / total for w in action_weights]
            node_weights = {node: weight for node, weight in zip(available_nodes, action_weights)}
        else:
            # 利用：选择最优动作
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                # 取得最大Q值的索引作为动作
                action_indices = torch.topk(q_values, len(available_nodes)).indices
                
                # 根据Q值计算节点权重
                q_values_np = q_values.cpu().numpy()
                # 对Q值做softmax，得到概率分布
                exp_q = np.exp(q_values_np - np.max(q_values_np))
                softmax_q = exp_q / np.sum(exp_q)
                
                node_weights = {available_nodes[i]: float(softmax_q[i]) for i in range(len(available_nodes))}
        
        # 衰减探索率
        if training and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return node_weights
        
    def store_experience(self, experience: RLExperience):
        """存储经验
        
        Args:
            experience: 经验对象
        """
        self.memory.add(experience)
        
    def train(self):
        """训练模型"""
        if len(self.memory) < self.batch_size or not self.initialized:
            return
        
        # 从经验回放缓冲区采样批次
        batch = self.memory.sample(self.batch_size)
        
        # 批次拆解为张量
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for exp in batch:
            states.append(self.preprocess_state(exp.state))
            
            # 获取动作索引
            action_weights = list(exp.action.node_weights.values())
            # 使用argmax选择权重最高的作为离散动作
            action_idx = np.argmax(action_weights)
            actions.append(action_idx)
            
            rewards.append(exp.reward)
            
            if exp.next_state:
                next_states.append(self.preprocess_state(exp.next_state))
            else:
                next_states.append(torch.zeros_like(states[-1]))
                
            dones.append(float(exp.done))
            
        # 转换为批次张量
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)
        
        # 计算当前Q值
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q
            
        # 计算损失
        loss = nn.MSELoss()(q_values, target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 增加训练步数
        self.training_steps += 1
        
        # 目标网络更新
        if self.training_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            logger.info(f"目标网络更新，当前训练步数: {self.training_steps}")
            
        # 模型保存
        if self.training_steps % self.model_save_interval == 0:
            self.save_model()
            
        return loss.item()
    
    def calculate_reward(self, 
                       selected_node: str, 
                       node_scores: Dict[str, float], 
                       resource_info: Dict[str, Dict[str, float]]) -> float:
        """计算奖励
        
        Args:
            selected_node: 选择的节点
            node_scores: 节点得分字典
            resource_info: 资源信息
            
        Returns:
            奖励值
        """
        # 基于得分计算的奖励
        score_reward = node_scores.get(selected_node, 0.0)
        
        # 计算资源均衡奖励
        balance_reward = 0.0
        if selected_node in resource_info:
            node_resource = resource_info[selected_node]
            cpu_usage = node_resource.get('cpu', 0.5)
            memory_usage = node_resource.get('memory', 0.5)
            
            # 计算资源使用的标准差，标准差越小表示资源使用越均衡
            std_dev = np.std([cpu_usage, memory_usage])
            # 标准差越小，奖励越高
            balance_reward = 1.0 - min(std_dev, 1.0)
        
        # 总奖励 = 得分奖励 + 均衡奖励
        total_reward = 0.7 * score_reward + 0.3 * balance_reward
        
        return total_reward
    
    def save_model(self):
        """保存模型"""
        if not self.initialized:
            return
            
        try:
            model_info = RLModelInfo(
                model_name=f"{self.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                algorithm=self.algorithm,
                total_training_steps=self.training_steps,
                last_updated=time.time(),
                average_reward=0.0,  # 暂不计算平均奖励
                hyperparameters={
                    "learning_rate": self.learning_rate,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "batch_size": self.batch_size
                }
            )
            
            # 保存模型权重
            model_path = os.path.join(self.model_path, f"{model_info.model_name}.pt")
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_steps': self.training_steps,
                'epsilon': self.epsilon
            }, model_path)
            
            # 保存模型信息
            info_path = os.path.join(self.model_path, f"{model_info.model_name}_info.pkl")
            joblib.dump(model_info.model_dump(), info_path)
            
            logger.info(f"模型已保存，路径: {model_path}")
        except Exception as e:
            logger.error(f"保存模型出错: {str(e)}")
    
    def load_model(self):
        """加载最新模型"""
        try:
            # 查找最新模型
            model_files = [f for f in os.listdir(self.model_path) if f.endswith('.pt')]
            if not model_files:
                logger.info("未找到现有模型，使用初始模型")
                return False
                
            # 按文件修改时间排序，取最新的
            model_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.model_path, x)), reverse=True)
            latest_model = os.path.join(self.model_path, model_files[0])
            
            # 根据模型文件名获取对应的信息文件
            model_name = os.path.splitext(model_files[0])[0]
            info_path = os.path.join(self.model_path, f"{model_name}_info.pkl")
            
            if os.path.exists(info_path):
                # 加载模型信息
                model_info = joblib.load(info_path)
                logger.info(f"加载模型信息: {model_info}")
                
                # 获取状态维度和动作维度
                self.algorithm = model_info.get('algorithm', self.algorithm)
                self.training_steps = model_info.get('total_training_steps', 0)
                
                # 如果获取不到维度信息，延迟初始化
                state_dim = model_info.get('state_dim')
                action_dim = model_info.get('action_dim')
                
                if state_dim and action_dim:
                    # 初始化网络
                    self._initialize_networks(state_dim, action_dim)
            
            # 加载模型权重
            checkpoint = torch.load(latest_model, map_location=self.device)
            
            # 如果网络还未初始化但有状态字典，可以通过状态字典推断网络结构
            if not self.initialized and 'q_network_state_dict' in checkpoint:
                # 尝试通过状态字典中的形状推断网络维度
                # 这是启发式的，可能不适用于所有网络结构
                first_layer = next(iter(checkpoint['q_network_state_dict'].items()))
                if 'weight' in first_layer[0]:
                    state_dim = first_layer[1].shape[1]
                    
                    # 找最后一层来推断动作维度
                    action_layer = None
                    for name, param in checkpoint['q_network_state_dict'].items():
                        if 'weight' in name:
                            action_layer = param
                    
                    if action_layer is not None:
                        action_dim = action_layer.shape[0]
                        
                        # 初始化网络
                        self._initialize_networks(state_dim, action_dim)
            
            # 尝试加载状态字典
            if self.initialized:
                self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.training_steps = checkpoint.get('training_steps', self.training_steps)
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                
                logger.info(f"模型加载成功，当前训练步数: {self.training_steps}")
                return True
            else:
                logger.warning("无法加载模型，网络未初始化")
                return False
                
        except Exception as e:
            logger.warning(f"加载模型出错: {str(e)}")
            return False 