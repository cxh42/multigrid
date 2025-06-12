import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import wandb
from collections import deque
import math

# Import multigrid environment
import gym
from envs import gym_multigrid
from envs.gym_multigrid import multigrid_envs

class SimpleMultiGridNet(nn.Module):
    """简化的多智能体网络"""
    
    def __init__(self, image_shape, n_actions):
        super().__init__()
        
        # 图像处理
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 计算卷积输出维度
        h, w = image_shape[0], image_shape[1]
        conv_out_h = ((h - 3) + 1 - 3) + 1  # 两次3x3卷积
        conv_out_w = ((w - 3) + 1 - 3) + 1
        conv_out_size = 64 * conv_out_h * conv_out_w
        
        # 特征融合
        self.feature_fc = nn.Sequential(
            nn.Linear(conv_out_size + 4, 128),  # 图像特征 + 方向one-hot
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 策略头
        self.actor = nn.Linear(64, n_actions)
        
        # 价值头
        self.critic = nn.Linear(64, 1)
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """初始化权重"""
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
            torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.orthogonal_(m.weight, np.sqrt(2))
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
        
    def forward(self, obs):
        """前向传播"""
        # 处理图像
        image = obs['image'].float()
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        # 转换为 (B, C, H, W)
        image = image.permute(0, 3, 1, 2)
        
        # 图像特征
        img_features = self.image_conv(image)
        
        # 处理方向
        direction = obs['direction']
        if isinstance(direction, np.ndarray):
            direction = torch.tensor(direction).long()
        elif not isinstance(direction, torch.Tensor):
            direction = torch.tensor([direction]).long()
        else:
            direction = direction.long()
        
        if len(direction.shape) == 0:
            direction = direction.unsqueeze(0)
        
        direction = direction.to(image.device)
        dir_onehot = F.one_hot(direction, num_classes=4).float()
        
        # 合并特征
        combined_features = torch.cat([img_features, dir_onehot], dim=1)
        features = self.feature_fc(combined_features)
        
        # 输出动作和价值
        logits = self.actor(features)
        value = self.critic(features)
        
        return logits, value

class PPOAgent:
    """单个PPO智能体"""
    
    def __init__(self, image_shape, n_actions, config, device):
        self.device = device
        self.config = config
        
        # 网络
        self.network = SimpleMultiGridNet(image_shape, n_actions).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)
        
        # PPO参数
        self.clip_coef = getattr(config, 'clip_coef', 0.2)
        self.ent_coef = getattr(config, 'ent_coef', 0.01)
        self.vf_coef = getattr(config, 'vf_coef', 0.5)
        self.max_grad_norm = getattr(config, 'max_grad_norm', 0.5)
        self.gamma = getattr(config, 'gamma', 0.99)
        self.gae_lambda = getattr(config, 'gae_lambda', 0.95)
        
        # 经验缓冲
        self.reset_buffer()
        
    def reset_buffer(self):
        """重置经验缓冲"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def get_action_and_value(self, obs):
        """获取动作和价值"""
        with torch.no_grad():
            obs_tensor = {
                'image': torch.tensor(obs['image']).float().to(self.device),
                'direction': torch.tensor(obs['direction']).long().to(self.device) if not isinstance(obs['direction'], torch.Tensor) else obs['direction'].long().to(self.device)
            }
            
            logits, value = self.network(obs_tensor)
            
            # 采样动作
            probs = Categorical(logits=logits)
            action = probs.sample()
            log_prob = probs.log_prob(action)
            
        return action.cpu().item(), log_prob.cpu().item(), value.cpu().item()
    
    def store_experience(self, state, action, reward, done, log_prob, value):
        """存储经验"""
        self.states.append(state)
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        
    def compute_gae(self, next_value):
        """计算广义优势估计"""
        advantages = []
        gae = 0.0
        
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_non_terminal = 1.0 - float(self.dones[i])
                next_val = float(next_value)
            else:
                next_non_terminal = 1.0 - float(self.dones[i + 1])
                next_val = float(self.values[i + 1])
                
            delta = float(self.rewards[i]) + self.gamma * next_val * next_non_terminal - float(self.values[i])
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            
        returns = [float(adv) + float(val) for adv, val in zip(advantages, self.values)]
        return advantages, returns
    
    def update(self, next_value):
        """更新网络"""
        if len(self.states) == 0:
            return {}
            
        advantages, returns = self.compute_gae(next_value)
        
        # 转换为张量，确保数据类型为float
        images = torch.stack([torch.tensor(s['image']).float() for s in self.states]).to(self.device)
        directions = torch.tensor([s['direction'] for s in self.states]).long().to(self.device)
        states_batch = {'image': images, 'direction': directions}
        
        actions_batch = torch.tensor(self.actions).long().to(self.device)
        old_log_probs_batch = torch.tensor(self.log_probs).float().to(self.device)
        advantages_batch = torch.tensor(advantages).float().to(self.device)
        returns_batch = torch.tensor(returns).float().to(self.device)
        
        # 标准化优势
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
        
        # 多轮更新
        total_loss = 0
        for _ in range(getattr(self.config, 'update_epochs', 4)):
            # 前向传播
            logits, values = self.network(states_batch)
            values = values.squeeze()
            
            probs = Categorical(logits=logits)
            new_log_probs = probs.log_prob(actions_batch)
            entropy = probs.entropy().mean()
            
            # PPO损失
            ratio = torch.exp(new_log_probs - old_log_probs_batch)
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * advantages_batch
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = F.mse_loss(values, returns_batch)
            
            # 总损失
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # 清空缓冲
        self.reset_buffer()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss / getattr(self.config, 'update_epochs', 4)
        }
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

class MultiAgentPPO:
    """多智能体PPO元控制器"""
    
    def __init__(self, config, env, device, debug=False):
        self.config = config
        self.env = env
        self.device = device
        self.debug = debug
        self.n_agents = env.n_agents
        self.total_steps = 0
        
        # 获取观察空间
        obs = env.reset()
        if isinstance(obs, dict) and isinstance(obs.get('image'), list):
            image_shape = np.array(obs['image'][0]).shape
        else:
            image_shape = (7, 7, 3)  # 默认值
        
        # 创建智能体
        self.agents = []
        for i in range(self.n_agents):
            agent = PPOAgent(image_shape, 7, config, device)  # 7个动作
            self.agents.append(agent)
        
        # 跟踪变量
        self.episode_rewards = deque(maxlen=100)
        self.best_avg_reward = -float('inf')
        self.static_counters = [0] * self.n_agents
        self.last_positions = [None] * self.n_agents
        self.visited_positions = [set() for _ in range(self.n_agents)]
        
        # 创建模型目录
        os.makedirs('model5', exist_ok=True)
        
    def get_agent_state(self, state, agent_id):
        """提取智能体特定的状态"""
        if isinstance(state, dict):
            if isinstance(state.get('image'), list):
                return {
                    'image': state['image'][agent_id],
                    'direction': state['direction'][agent_id]
                }
            else:
                return {
                    'image': state['image'],
                    'direction': state['direction']
                }
        else:
            return state[agent_id]
    
    def compute_reward(self, env, agent_id, prev_pos, curr_pos):
        """计算增强奖励"""
        reward = 0
        
        # 寻找目标
        goal_pos = None
        try:
            for x in range(env.grid.width):
                for y in range(env.grid.height):
                    cell = env.grid.get(x, y)
                    if cell and hasattr(cell, 'type') and cell.type == 'goal':
                        goal_pos = np.array([x, y])
                        break
                if goal_pos is not None:
                    break
                    
            if goal_pos is not None and curr_pos is not None and prev_pos is not None:
                # 距离奖励
                prev_dist = np.linalg.norm(prev_pos - goal_pos)
                curr_dist = np.linalg.norm(curr_pos - goal_pos)
                
                if curr_dist < prev_dist:
                    reward += 0.5  # 靠近目标
                elif curr_dist > prev_dist:
                    reward -= 0.2  # 远离目标
                
                # 距离反比奖励
                reward += 1.0 / (1.0 + curr_dist)
                
                # 到达目标大奖励
                if curr_dist < 1.5:
                    reward += 5.0
        except:
            pass
        
        # 静止惩罚
        if prev_pos is not None and curr_pos is not None:
            if np.array_equal(prev_pos, curr_pos):
                self.static_counters[agent_id] += 1
                reward -= 0.02 * self.static_counters[agent_id]
            else:
                self.static_counters[agent_id] = 0
        
        # 探索奖励
        if curr_pos is not None:
            pos_tuple = tuple(curr_pos)
            if pos_tuple not in self.visited_positions[agent_id]:
                self.visited_positions[agent_id].add(pos_tuple)
                reward += 0.1
        
        # 时间惩罚
        reward -= 0.001
        
        # 严重静止惩罚
        if self.static_counters[agent_id] > 20:
            reward -= 1.0
        
        return reward
    
    def run_one_episode(self, episode):
        """运行一个episode"""
        state = self.env.reset()
        
        # 调试信息
        if episode == 0:
            print(f"环境观察格式:")
            print(f"  类型: {type(state)}")
            if isinstance(state, dict):
                print(f"  键: {state.keys()}")
                if 'image' in state:
                    print(f"  图像类型: {type(state['image'])}")
                    if isinstance(state['image'], list):
                        print(f"  图像长度: {len(state['image'])}, 单个形状: {np.array(state['image'][0]).shape}")
                    else:
                        print(f"  图像形状: {np.array(state['image']).shape}")
        
        episode_reward = [0] * self.n_agents
        episode_length = 0
        done = False
        
        # 重置统计
        self.static_counters = [0] * self.n_agents
        self.last_positions = [None] * self.n_agents
        self.visited_positions = [set() for _ in range(self.n_agents)]
        
        while not done and episode_length < 100:
            actions = []
            log_probs = []
            values = []
            
            # 获取当前位置
            current_positions = []
            for i in range(self.n_agents):
                pos = getattr(self.env, 'agent_pos', [None] * self.n_agents)[i]
                current_positions.append(pos.copy() if pos is not None else None)
            
            # 每个智能体选择动作
            for i in range(self.n_agents):
                agent_state = self.get_agent_state(state, i)
                action, log_prob, value = self.agents[i].get_action_and_value(agent_state)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
            
            # 执行动作
            next_state, env_rewards, done, _ = self.env.step(actions)
            
            # 获取新位置
            new_positions = []
            for i in range(self.n_agents):
                pos = getattr(self.env, 'agent_pos', [None] * self.n_agents)[i]
                new_positions.append(pos.copy() if pos is not None else None)
            
            # 计算奖励并存储经验
            for i in range(self.n_agents):
                # 增强奖励
                enhanced_reward = self.compute_reward(
                    self.env, i, self.last_positions[i], current_positions[i]
                )
                
                # 添加环境奖励
                if isinstance(env_rewards, list):
                    enhanced_reward += env_rewards[i]
                else:
                    enhanced_reward += env_rewards
                
                # 存储经验
                agent_state = self.get_agent_state(state, i)
                self.agents[i].store_experience(
                    agent_state, actions[i], enhanced_reward, done, log_probs[i], values[i]
                )
                
                episode_reward[i] += enhanced_reward
            
            # 更新位置
            self.last_positions = current_positions
            state = next_state
            episode_length += 1
            self.total_steps += 1
        
        # Episode结束，更新网络
        for i in range(self.n_agents):
            final_state = self.get_agent_state(state, i)
            with torch.no_grad():
                obs_tensor = {
                    'image': torch.tensor(final_state['image']).float().to(self.device),
                    'direction': torch.tensor(final_state['direction']).long().to(self.device)
                }
                _, next_value = self.agents[i].network(obs_tensor)
                next_value = next_value.cpu().item()
            
            # 更新智能体
            self.agents[i].update(next_value)
        
        total_reward = sum(episode_reward)
        self.episode_rewards.append(total_reward)
        
        return total_reward, episode_length
    
    def train(self):
        """训练主循环"""
        print("开始训练多智能体PPO...")
        
        # 初始化wandb
        if not self.debug:
            wandb.init(
                project=getattr(self.config, 'wandb_project', 'multigrid-ppo-v5'),
                config=vars(self.config),
                name=f"MultiAgent_PPO_v5_{int(time.time())}"
            )
        
        for episode in range(getattr(self.config, 'n_episodes', 10000)):
            total_reward, episode_length = self.run_one_episode(episode)
            
            # 每100个episode记录
            if episode % 100 == 0:
                avg_reward = np.mean(list(self.episode_rewards)) if self.episode_rewards else 0
                print(f"Episode {episode}, 平均奖励(最近100): {avg_reward:.2f}, "
                      f"当前奖励: {total_reward:.2f}, Episode长度: {episode_length}")
                
                if not self.debug:
                    wandb.log({
                        'episode': episode,
                        'avg_reward_100': avg_reward,
                        'current_reward': total_reward,
                        'episode_length': episode_length,
                        'total_steps': self.total_steps
                    })
                
                # 保存最佳模型
                if len(self.episode_rewards) >= 100:
                    if avg_reward > self.best_avg_reward:
                        self.best_avg_reward = avg_reward
                        print(f"新的最佳平均奖励: {avg_reward:.2f}, 保存模型...")
                        self.save_best_model()
            
            # 定期保存检查点
            if episode % 1000 == 0 and episode > 0:
                self.save_checkpoint(episode)
        
        print("训练完成！")
        if not self.debug:
            wandb.finish()
    
    def save_best_model(self):
        """保存最佳模型"""
        for i, agent in enumerate(self.agents):
            model_path = f'model5/best_agent_{i}.pth'
            agent.save_model(model_path)
        print("最佳模型已保存到 model5/ 文件夹")
    
    def save_checkpoint(self, episode):
        """保存检查点"""
        for i, agent in enumerate(self.agents):
            checkpoint_path = f'model5/checkpoint_agent_{i}_episode_{episode}.pth'
            agent.save_model(checkpoint_path)
        print(f"检查点已保存: episode {episode}")

def main():
    # 配置
    class Config:
        domain = "MultiGrid-Cluttered-Fixed-15x15"
        n_episodes = 50000
        learning_rate = 2.5e-4
        gamma = 0.99
        gae_lambda = 0.95
        clip_coef = 0.2
        ent_coef = 0.01
        vf_coef = 0.5
        max_grad_norm = 0.5
        update_epochs = 4
        wandb_project = "multigrid-ppo-v5"
        seed = 42
    
    config = Config()
    
    # 设置随机种子
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境
    env = gym.make(config.domain)
    print(f"环境: {config.domain}, 智能体数量: {env.n_agents}")
    
    # 创建训练器
    trainer = MultiAgentPPO(config, env, device, debug=False)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()