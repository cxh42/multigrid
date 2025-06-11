import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.categorical import Categorical
import argparse
import matplotlib.pyplot as plt

# Import needed to trigger env registration
from envs import gym_multigrid
from envs.gym_multigrid import multigrid_envs


class MultiGridPPOAgent(nn.Module):
    def __init__(self, n_actions=7):
        super().__init__()
        
        # 图像处理网络 - 适配5x5图像
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 保持 5x5
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), # 保持 5x5  
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),    # 降到 1x1
            nn.Flatten()                     # 32维特征
        )
        
        # 方向编码 (0-3: 4个方向)
        self.direction_embed = nn.Embedding(4, 8)
        
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(32 + 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Actor和Critic头
        self.actor = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)
        
    def forward(self, obs):
        """前向传播"""
        # 提取图像和方向
        image = obs['image']  # [batch, H, W, C] 或 [H, W, C]
        direction = obs['direction']  # [batch] 或 [1]
        
        # 转换图像格式：(B, H, W, C) -> (B, C, H, W)
        if len(image.shape) == 4:
            image = image.permute(0, 3, 1, 2)
        elif len(image.shape) == 3:
            image = image.permute(2, 0, 1).unsqueeze(0)
        
        # 确保方向是正确的形状
        if len(direction.shape) > 1:
            direction = direction.squeeze(-1)  # 移除多余维度
        if len(direction.shape) == 0:
            direction = direction.unsqueeze(0)  # 添加batch维度
        
        # 图像特征提取
        image_features = self.image_conv(image.float())  # [batch, 32]
        
        # 方向特征提取  
        direction_features = self.direction_embed(direction.long())  # [batch, 8]
        
        # 确保两个tensor的batch维度一致
        batch_size = image_features.shape[0]
        if direction_features.shape[0] != batch_size:
            if direction_features.shape[0] == 1:
                direction_features = direction_features.repeat(batch_size, 1)
            else:
                raise ValueError(f"Batch size mismatch: image {batch_size}, direction {direction_features.shape[0]}")
        
        # 合并特征
        combined = torch.cat([image_features, direction_features], dim=-1)  # [batch, 40]
        shared_features = self.shared(combined)
        
        # 输出
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        
        return action_logits, value.squeeze(-1)
    
    def get_action_and_value(self, obs, action=None):
        """获取动作和值"""
        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value


def preprocess_obs(obs, device):
    """预处理观察，转换为tensor"""
    # multigrid返回的格式：{'image': [numpy_array], 'direction': [int]}
    # 需要从列表中提取实际的数组和值
    image = obs['image'][0]  # 提取numpy数组，形状: [5, 5, 3]
    direction = obs['direction'][0]  # 提取int值
    
    return {
        'image': torch.FloatTensor(image).to(device),  # [5, 5, 3]
        'direction': torch.LongTensor([direction]).to(device)  # [1]
    }


def collect_rollout(env, agent, device, n_steps=128):
    """收集一轮经验"""
    observations = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    dones = []
    
    # 重置环境
    obs = env.reset()
    
    for step in range(n_steps):
        # 预处理观察
        obs_tensor = preprocess_obs(obs, device)
        
        # 获取动作
        with torch.no_grad():
            action, log_prob, entropy, value = agent.get_action_and_value(obs_tensor)
        
        # 存储数据
        observations.append(obs)
        actions.append(action.item())
        log_probs.append(log_prob.item())
        values.append(value.item())
        
        # 执行动作（注意：需要传入列表格式）
        next_obs, reward, done, info = env.step([action.item()])
        
        # 处理奖励（multigrid返回列表格式）
        if isinstance(reward, list):
            reward = reward[0]
        
        rewards.append(reward)
        dones.append(done)
        
        if done:
            obs = env.reset()
        else:
            obs = next_obs
    
    return observations, actions, log_probs, values, rewards, dones


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """计算GAE优势"""
    advantages = []
    returns = []
    
    advantage = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        next_non_terminal = 1 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantage = delta + gamma * gae_lambda * next_non_terminal * advantage
        
        advantages.insert(0, advantage)
        returns.insert(0, advantage + values[t])
    
    return advantages, returns


def ppo_update(agent, optimizer, observations, actions, old_log_probs, advantages, returns, device, 
               n_epochs=4, batch_size=64, clip_coef=0.2):
    """PPO更新"""
    
    # 转换为tensor
    actions_tensor = torch.LongTensor(actions).to(device)
    old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(device)
    advantages_tensor = torch.FloatTensor(advantages).to(device)
    returns_tensor = torch.FloatTensor(returns).to(device)
    
    # 标准化优势
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
    
    # 多轮更新
    for epoch in range(n_epochs):
        # 随机采样
        indices = torch.randperm(len(observations))
        
        for start in range(0, len(observations), batch_size):
            end = min(start + batch_size, len(observations))
            batch_indices = indices[start:end]
            
            # 预处理batch观察
            batch_obs_list = []
            for i in batch_indices:
                obs_tensor = preprocess_obs(observations[i], device)
                batch_obs_list.append(obs_tensor)
            
            # 构建batch tensor - 注意这里的正确方式
            batch_images = torch.stack([obs['image'] for obs in batch_obs_list])  # [batch, H, W, C]
            batch_directions = torch.stack([obs['direction'] for obs in batch_obs_list])  # [batch, 1]
            
            # 注意：direction需要展平为1D
            batch_directions = batch_directions.squeeze(-1)  # [batch] 
            
            batch_obs = {
                'image': batch_images,
                'direction': batch_directions
            }
            
            batch_actions = actions_tensor[batch_indices]
            batch_old_log_probs = old_log_probs_tensor[batch_indices]
            batch_advantages = advantages_tensor[batch_indices]
            batch_returns = returns_tensor[batch_indices]
            
            # 前向传播
            _, new_log_probs, entropy, new_values = agent.get_action_and_value(batch_obs, batch_actions)
            
            # 计算比率
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            
            # PPO损失
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 值函数损失
            value_loss = 0.5 * (new_values - batch_returns).pow(2).mean()
            
            # 熵奖励
            entropy_loss = -0.01 * entropy.mean()
            
            # 总损失
            total_loss = policy_loss + value_loss + entropy_loss
            
            # 更新
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()


def train_ppo(env_name, n_episodes=1000, n_steps=128, save_path=None):
    """训练PPO智能体"""
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境
    env = gym.make(env_name)
    print(f"训练环境: {env_name}")
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    
    # 创建智能体
    agent = MultiGridPPOAgent(n_actions=7).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    
    # 训练记录
    episode_rewards = []
    episode_lengths = []
    
    print("开始训练...")
    
    for episode in range(n_episodes):
        # 收集经验
        observations, actions, log_probs, values, rewards, dones = collect_rollout(
            env, agent, device, n_steps)
        
        # 计算优势
        advantages, returns = compute_gae(rewards, values, dones)
        
        # PPO更新
        ppo_update(agent, optimizer, observations, actions, log_probs, 
                  advantages, returns, device)
        
        # 记录统计
        episode_reward = sum(rewards)
        episode_length = len(rewards)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 打印进度
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Episode {episode:4d} | Avg Reward: {avg_reward:6.2f} | Avg Length: {avg_length:6.1f}")
    
    # 保存模型
    if save_path:
        torch.save(agent.state_dict(), save_path)
        print(f"模型已保存到: {save_path}")
    
    env.close()
    return agent, episode_rewards


def test_agent(env_name, model_path, n_episodes=5):
    """测试训练好的智能体"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建环境和智能体
    env = gym.make(env_name)
    agent = MultiGridPPOAgent(n_actions=7).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()
    
    print(f"测试环境: {env_name}")
    
    for episode in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            obs_tensor = preprocess_obs(obs, device)
            
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            
            obs, reward, done, info = env.step([action.item()])
            
            if isinstance(reward, list):
                reward = reward[0]
            
            total_reward += reward
            steps += 1
            
            if done or steps > 200:
                break
        
        print(f"测试Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MultiGrid-Empty-5x5-Single-v0")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--test", type=str, default=None, help="测试模式，提供模型路径")
    args = parser.parse_args()
    
    if args.test:
        test_agent(args.env, args.test)
    else:
        model_path = f"ppo_{args.env.replace('-', '_')}.pth"
        agent, rewards = train_ppo(args.env, args.episodes, save_path=model_path)
        
        print(f"训练完成！最后10轮平均奖励: {np.mean(rewards[-10:]):.2f}")
        
        # 简单的性能图
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # 移动平均
        window = 50
        if len(rewards) > window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.subplot(1, 2, 2)
            plt.plot(moving_avg)
            plt.title(f'Moving Average ({window} episodes)')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
        
        plt.tight_layout()
        plt.savefig(f'training_results_{args.env.replace("-", "_")}.png')
        print("训练曲线已保存")