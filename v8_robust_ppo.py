"""
v8极简高速版多智能体PPO - 回归基础

特性：
1. 单环境训练，最简设计
2. 每100个episode log一次
3. 滑动窗口最佳模型保存
4. 每1000episode保存检查点
5. 极简代码，最高速度
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.categorical import Categorical
import argparse
import os
import time
import json
import warnings
warnings.filterwarnings("ignore")

from envs import gym_multigrid
from envs.gym_multigrid import multigrid_envs

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SimplePPOAgent(nn.Module):
    """极简PPO网络"""
    
    def __init__(self, n_actions=7):
        super().__init__()
        
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.direction_embed = nn.Embedding(4, 8)
        
        self.shared = nn.Sequential(
            nn.Linear(72, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        self.actor = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)
        
        # 简单初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 0.5)
                m.bias.data.zero_()
                
    def forward(self, obs):
        image = obs['image']
        direction = obs['direction']
        
        if len(image.shape) == 3:
            image = image.permute(2, 0, 1).unsqueeze(0)
        elif len(image.shape) == 4:
            image = image.permute(0, 3, 1, 2)
        
        if len(direction.shape) == 0:
            direction = direction.unsqueeze(0)
        elif len(direction.shape) > 1:
            direction = direction.squeeze()
        
        image_features = self.image_conv(image.float())
        direction_features = self.direction_embed(direction.long())
        
        if image_features.shape[0] != direction_features.shape[0]:
            direction_features = direction_features.repeat(image_features.shape[0], 1)
        
        combined = torch.cat([image_features, direction_features], dim=-1)
        shared_features = self.shared(combined)
        
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        
        return action_logits, value.squeeze(-1)
    
    def get_action_and_value(self, obs, action=None):
        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value


class SimpleController:
    """极简控制器"""
    
    def __init__(self, env_name, n_agents, device, lr=1e-4):  # 提高学习率
        self.env_name = env_name
        self.n_agents = n_agents
        self.device = device
        
        # 创建智能体
        self.agents = []
        self.optimizers = []
        
        for i in range(n_agents):
            agent = SimplePPOAgent().to(device)
            optimizer = optim.Adam(agent.parameters(), lr=lr)
            self.agents.append(agent)
            self.optimizers.append(optimizer)
        
        # 单个环境
        self.env = gym.make(env_name)
        
        # 奖励塑形状态
        self.prev_distances = [None] * n_agents
        self.prev_positions = [None] * n_agents
        self.stationary_count = [0] * n_agents
        
        # 滑动窗口最佳模型
        self.performance_window = []
        self.window_size = 100
        self.best_avg_performance = float('-inf')
        
        print(f"极简控制器: {n_agents}智能体, 学习率{lr}, 设备{device}")
    
    def get_actions(self, obs):
        """获取动作"""
        actions = []
        log_probs = []
        values = []
        
        for i in range(self.n_agents):
            agent_obs = {
                'image': torch.FloatTensor(obs['image'][i]).to(self.device),
                'direction': torch.LongTensor([obs['direction'][i]]).to(self.device)
            }
            
            with torch.no_grad():
                action, log_prob, entropy, value = self.agents[i].get_action_and_value(agent_obs)
            
            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value.item())
        
        return actions, log_probs, values
    
    def get_goal_position(self):
        """获取目标位置"""
        try:
            for i in range(self.env.width):
                for j in range(self.env.height):
                    cell = self.env.grid.get(i, j)
                    if cell and cell.type == 'goal':
                        return np.array([i, j])
            return np.array([13, 13])
        except:
            return np.array([13, 13])
    
    def shape_rewards(self, agent_positions, original_rewards, actions):
        """v7风格的奖励塑形"""
        shaped_rewards = []
        goal_pos = self.get_goal_position()
        
        for i in range(self.n_agents):
            pos = np.array(agent_positions[i])
            action = actions[i]
            
            # 1. 触碰目标 - 大奖励
            if original_rewards[i] > 0:
                shaped_rewards.append(5.0)
                self.prev_distances[i] = None  # 重置
                self.prev_positions[i] = None
                self.stationary_count[i] = 0
                continue
            
            # 重置奖励
            reward = 0.0
            
            # 2. 距离奖励 - 关键引导
            current_dist = np.linalg.norm(pos - goal_pos)
            if self.prev_distances[i] is not None:
                dist_change = self.prev_distances[i] - current_dist
                reward += dist_change * 0.2  # 提高距离奖励系数
            self.prev_distances[i] = current_dist
            
            # 3. 静止惩罚
            if self.prev_positions[i] is not None:
                if np.array_equal(pos, self.prev_positions[i]):
                    self.stationary_count[i] += 1
                    if self.stationary_count[i] > 3:
                        reward -= 0.05
                else:
                    self.stationary_count[i] = 0
                    reward += 0.02  # 移动奖励
            
            # 4. 行动奖励
            if action == 2:  # forward
                reward += 0.02
            elif action in [0, 1]:  # turn
                reward += 0.01
                
            # 5. 限制奖励范围
            reward = max(reward, -0.2)
            
            shaped_rewards.append(reward)
            self.prev_positions[i] = pos.copy()
            
        return shaped_rewards
    
    def run_episode(self):
        """运行一个episode"""
        obs = self.env.reset()
        
        # 重置奖励塑形状态
        self.prev_distances = [None] * self.n_agents
        self.prev_positions = [None] * self.n_agents
        self.stationary_count = [0] * self.n_agents
        
        # 存储轨迹
        trajectories = [[] for _ in range(self.n_agents)]
        
        episode_rewards = [0] * self.n_agents
        goal_touches = 0
        
        for step in range(500):  # 最大步数
            actions, log_probs, values = self.get_actions(obs)
            next_obs, rewards, done, info = self.env.step(actions)
            
            # v7风格奖励塑形
            agent_positions = [self.env.agent_pos[i] for i in range(self.n_agents)]
            shaped_rewards = self.shape_rewards(agent_positions, rewards, actions)
            
            # 统计目标触碰
            for r in shaped_rewards:
                if r >= 9.5:
                    goal_touches += 1
            
            # 存储轨迹
            for i in range(self.n_agents):
                agent_obs = {
                    'image': torch.FloatTensor(obs['image'][i]).to(self.device),
                    'direction': torch.LongTensor([obs['direction'][i]]).to(self.device)
                }
                
                trajectories[i].append({
                    'obs': agent_obs,
                    'action': actions[i],
                    'log_prob': log_probs[i],
                    'value': values[i],
                    'reward': shaped_rewards[i],
                    'done': done
                })
                
                episode_rewards[i] += shaped_rewards[i]
            
            obs = next_obs
            
            if done:
                break
        
        return trajectories, episode_rewards, goal_touches
    
    def compute_gae(self, trajectory, gamma=0.99, gae_lambda=0.95):
        """计算GAE"""
        rewards = [t['reward'] for t in trajectory]
        values = [t['value'] for t in trajectory]
        dones = [t['done'] for t in trajectory]
        
        advantages = []
        returns = []
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            next_value = 0 if t == len(rewards) - 1 else values[t + 1]
            next_non_terminal = 1 - dones[t]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantage = delta + gamma * gae_lambda * next_non_terminal * advantage
            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[t])
        
        return advantages, returns
    
    def update_agent(self, agent_id, trajectory):
        """更新单个智能体"""
        if len(trajectory) == 0:
            return 0
        
        agent = self.agents[agent_id]
        optimizer = self.optimizers[agent_id]
        
        # 计算优势
        advantages, returns = self.compute_gae(trajectory)
        
        # 转换为张量
        observations = [t['obs'] for t in trajectory]
        actions = torch.LongTensor([t['action'] for t in trajectory]).to(self.device)
        old_log_probs = torch.FloatTensor([t['log_prob'] for t in trajectory]).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # 标准化优势
        if advantages_tensor.std() > 0:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # 批处理观察
        images = torch.stack([obs['image'] for obs in observations])
        directions = torch.stack([obs['direction'] for obs in observations]).squeeze()
        batch_obs = {'image': images, 'direction': directions}
        
        # PPO更新
        total_loss = 0
        for epoch in range(4):  # 增加更新轮数
            _, new_log_probs, entropy, new_values = agent.get_action_and_value(batch_obs, actions)
            
            # 计算损失
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages_tensor  # 更宽松的裁剪
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = 0.5 * (new_values - returns_tensor).pow(2).mean()
            entropy_loss = -0.02 * entropy.mean()  # 增加熵系数
            
            loss = policy_loss + value_loss + entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / 4
    
    def update_best_model(self, collective_reward):
        """更新最佳模型"""
        self.performance_window.append(collective_reward)
        
        if len(self.performance_window) > self.window_size:
            self.performance_window.pop(0)
        
        if len(self.performance_window) == self.window_size:
            avg_performance = np.mean(self.performance_window)
            if avg_performance > self.best_avg_performance:
                self.best_avg_performance = avg_performance
                self.save_models("models8/best_performance")
                return True
        return False
    
    def save_models(self, path_prefix):
        """保存模型"""
        os.makedirs(os.path.dirname(path_prefix) if os.path.dirname(path_prefix) else "models8", exist_ok=True)
        for i in range(self.n_agents):
            torch.save(self.agents[i].state_dict(), f"{path_prefix}_agent_{i}.pth")


def train_simple_ppo(
    env_name="MultiGrid-Cluttered-Fixed-15x15",
    n_episodes=100000,
    use_wandb=True,
    project_name="simple-ppo-v8",
    experiment_name=None
):
    """极简PPO训练"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 获取环境信息
    temp_env = gym.make(env_name)
    n_agents = temp_env.n_agents
    temp_env.close()
    
    print(f"极简训练: {env_name}, {n_agents}智能体, {n_episodes}轮, {device}")
    
    if experiment_name is None:
        experiment_name = f"simple_{env_name}_{n_agents}agents_{int(time.time())}"
    
    # WandB初始化
    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb.init(
                project=project_name,
                name=experiment_name,
                config={
                    "env_name": env_name,
                    "n_agents": n_agents,
                    "n_episodes": n_episodes,
                    "algorithm": "Simple PPO v8"
                }
            )
        except:
            use_wandb = False
    else:
        use_wandb = False
    
    # 创建控制器
    controller = SimpleController(env_name, n_agents, device, lr=1e-4)  # 提高学习率
    
    # 训练循环
    collective_rewards = []
    total_goal_touches = 0
    
    print("开始极简训练...")
    start_time = time.time()
    
    for episode in range(n_episodes):
        # 运行episode
        trajectories, episode_rewards, goal_touches = controller.run_episode()
        total_goal_touches += goal_touches
        
        # 更新智能体
        losses = []
        for i in range(n_agents):
            loss = controller.update_agent(i, trajectories[i])
            losses.append(loss)
        
        collective_reward = sum(episode_rewards)
        collective_rewards.append(collective_reward)
        
        # 更新最佳模型
        controller.update_best_model(collective_reward)
        
        # 每1000episode保存
        if episode % 1000 == 0 and episode > 0:
            controller.save_models(f"models8/{experiment_name}_ep{episode:06d}")
        
        # 计算统计
        if len(collective_rewards) >= 100:
            avg_collective = np.mean(collective_rewards[-100:])
        else:
            avg_collective = collective_reward
        
        # 每100episode记录
        if use_wandb:
            wandb.log({
                "episode": episode,
                "collective_reward": collective_reward,
                "avg_collective_reward_100": avg_collective,
                "goal_touches": goal_touches,
                "total_goal_touches": total_goal_touches,
                "avg_loss": np.mean(losses),
                "best_avg_performance": controller.best_avg_performance
            })
        
        # 每100episode输出
        if episode % 100 == 0:
            elapsed = time.time() - start_time
            speed = episode / elapsed * 3600 if elapsed > 0 else 0
            print(f"Ep {episode:6d} | 奖励: {avg_collective:7.2f} | "
                  f"最佳: {controller.best_avg_performance:7.2f} | "
                  f"目标: {goal_touches:2d} | 速度: {speed:.0f} ep/h")
    
    # 最终保存
    controller.save_models(f"models8/{experiment_name}_final")
    
    # 结果
    total_time = time.time() - start_time
    print(f"训练完成! 时间: {total_time/3600:.2f}小时")
    print(f"最佳平均: {controller.best_avg_performance:.2f}")
    print(f"总目标: {total_goal_touches}")
    
    # 保存结果
    results = {
        'collective_rewards': collective_rewards,
        'best_avg_performance': controller.best_avg_performance,
        'total_goal_touches': total_goal_touches,
        'total_time_hours': total_time / 3600
    }
    
    os.makedirs('models8', exist_ok=True)
    with open(f"models8/results_{experiment_name}.json", 'w') as f:
        json.dump(results, f)
    
    if use_wandb:
        wandb.finish()
    
    return controller, collective_rewards, experiment_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="极简PPO v8")
    parser.add_argument("--env", type=str, default="MultiGrid-Cluttered-Fixed-15x15")
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--project", type=str, default="simple-ppo-v8")
    parser.add_argument("--name", type=str, default=None)
    
    args = parser.parse_args()
    
    controller, collective_rewards, experiment_name = train_simple_ppo(
        env_name=args.env,
        n_episodes=args.episodes,
        use_wandb=not args.no_wandb,
        project_name=args.project,
        experiment_name=args.name
    )
    
    print(f"模型保存: models8/{experiment_name}_*")