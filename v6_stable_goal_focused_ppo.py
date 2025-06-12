"""
v4稳定版多智能体PPO - 修复训练崩溃问题

修复：
1. 降低学习率：3e-4 → 1e-4
2. 优化奖励设计，避免负奖励累积
3. 添加梯度监控和早停机制
4. 更保守的奖励塑形
5. 添加调试输出
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.categorical import Categorical
import argparse
import matplotlib.pyplot as plt
import os
import time
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from envs import gym_multigrid
from envs.gym_multigrid import multigrid_envs

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class StableRewardShaper:
    """稳定的奖励塑形器 - 避免负奖励累积"""
    
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.reset()
        
    def reset(self):
        self.prev_distances = [None] * self.n_agents
        self.prev_positions = [None] * self.n_agents
        self.stationary_count = [0] * self.n_agents
        
    def get_goal_position(self, env):
        """获取目标位置"""
        try:
            for i in range(env.width):
                for j in range(env.height):
                    cell = env.grid.get(i, j)
                    if cell and cell.type == 'goal':
                        return np.array([i, j])
            return np.array([13, 13])
        except:
            return np.array([13, 13])
    
    def shape_rewards(self, env, agent_positions, original_rewards, actions):
        """稳定的奖励塑形 - 避免大负值"""
        shaped_rewards = list(original_rewards)
        goal_pos = self.get_goal_position(env)
        
        for i in range(self.n_agents):
            pos = np.array(agent_positions[i])
            action = actions[i]
            
            # 1. 触碰目标 - 巨大正奖励
            if original_rewards[i] > 0:
                shaped_rewards[i] = 10.0  # 降低到10，更稳定
                print(f"🎯 智能体{i}触碰目标！奖励+10")
                continue
            
            # 重置shaped_rewards为0，避免累积原始负奖励
            shaped_rewards[i] = 0.0
            
            # 2. 距离奖励 - 更保守
            current_dist = np.linalg.norm(pos - goal_pos)
            if self.prev_distances[i] is not None:
                dist_change = self.prev_distances[i] - current_dist
                shaped_rewards[i] += dist_change * 0.2  # 降低到0.2
            self.prev_distances[i] = current_dist
            
            # 3. 静止惩罚 - 更温和且有上限
            if self.prev_positions[i] is not None:
                if np.array_equal(pos, self.prev_positions[i]):
                    self.stationary_count[i] += 1
                    # 只有连续静止才惩罚，且有上限
                    if self.stationary_count[i] > 3:
                        shaped_rewards[i] -= min(0.05, self.stationary_count[i] * 0.01)
                else:
                    self.stationary_count[i] = 0
                    shaped_rewards[i] += 0.01  # 移动奖励
                    
            # 4. 移动奖励 - 鼓励行动
            if action == 2:  # forward
                shaped_rewards[i] += 0.02
            elif action in [0, 1]:  # turn
                shaped_rewards[i] += 0.005
                
            # 5. 去掉时间惩罚，避免累积负值
            
            # 6. 确保奖励不会太负
            shaped_rewards[i] = max(shaped_rewards[i], -0.2)
            
            self.prev_positions[i] = pos.copy()
            
        return shaped_rewards


class SimplePPOAgent(nn.Module):
    """简化的PPO网络"""
    
    def __init__(self, n_actions=7):
        super().__init__()
        
        # 图像处理
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # 方向嵌入
        self.direction_embed = nn.Embedding(4, 8)
        
        # 共享网络
        self.shared = nn.Sequential(
            nn.Linear(64 + 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # 输出头
        self.actor = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)
        
        # 初始化权重 - 更保守
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, 0.5)
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.orthogonal_(module.weight, 1.0)
            
    def forward(self, obs):
        image = obs['image']
        direction = obs['direction']
        
        # 处理图像
        if len(image.shape) == 4:
            image = image.permute(0, 3, 1, 2)
        elif len(image.shape) == 3:
            image = image.permute(2, 0, 1).unsqueeze(0)
        
        # 处理方向
        if len(direction.shape) > 1:
            direction = direction.squeeze(-1)
        if len(direction.shape) == 0:
            direction = direction.unsqueeze(0)
        
        # 特征提取
        image_features = self.image_conv(image.float())
        direction_features = self.direction_embed(direction.long())
        
        # 确保维度匹配
        batch_size = image_features.shape[0]
        if direction_features.shape[0] != batch_size:
            direction_features = direction_features.repeat(batch_size, 1)
        
        # 合并特征
        combined = torch.cat([image_features, direction_features], dim=-1)
        shared_features = self.shared(combined)
        
        # 输出
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        
        return action_logits, value.squeeze(-1)
    
    def get_action_and_value(self, obs, action=None):
        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value


class StablePPOController:
    """稳定版PPO控制器"""
    
    def __init__(self, env_name, n_agents, device, lr=1e-4, n_parallel_envs=4):  # 降低学习率
        self.env_name = env_name
        self.n_agents = n_agents
        self.device = device
        self.n_parallel_envs = n_parallel_envs
        
        # 创建智能体和优化器
        self.agents = []
        self.optimizers = []
        
        for i in range(n_agents):
            agent = SimplePPOAgent(n_actions=7).to(device)
            optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
            
            self.agents.append(agent)
            self.optimizers.append(optimizer)
        
        # 创建环境
        self.envs = [gym.make(env_name) for _ in range(n_parallel_envs)]
        
        # 稳定的奖励塑形器
        self.reward_shapers = [StableRewardShaper(n_agents) for _ in range(n_parallel_envs)]
        
        # 训练监控
        self.entropy_coef = 0.01
        self.performance_history = []
        self.gradient_norms = []
        
        self.reset_buffers()
        
        print(f"✅ 创建了 {n_agents} 个稳定PPO智能体")
        print(f"✅ 学习率: {lr} (降低以提高稳定性)")
        print(f"✅ 使用 {n_parallel_envs} 个并行环境")
        print(f"✅ 设备: {device}")
    
    def reset_buffers(self):
        """重置经验缓冲区"""
        self.buffers = []
        for i in range(self.n_agents):
            self.buffers.append({
                'observations': [],
                'actions': [],
                'log_probs': [],
                'values': [],
                'rewards': [],
                'dones': []
            })
    
    def get_parallel_actions(self, obs_list):
        """获取并行环境的动作"""
        all_actions = []
        all_log_probs = []
        all_values = []
        all_individual_obs = []
        
        for env_idx, obs in enumerate(obs_list):
            env_actions = []
            env_log_probs = []
            env_values = []
            env_individual_obs = []
            
            for i in range(self.n_agents):
                agent_obs = {
                    'image': torch.FloatTensor(obs['image'][i]).to(self.device),
                    'direction': torch.LongTensor([obs['direction'][i]]).to(self.device)
                }
                
                with torch.no_grad():
                    action, log_prob, entropy, value = self.agents[i].get_action_and_value(agent_obs)
                
                env_actions.append(action.item())
                env_log_probs.append(log_prob.item())
                env_values.append(value.item())
                env_individual_obs.append(agent_obs)
            
            all_actions.append(env_actions)
            all_log_probs.append(env_log_probs)
            all_values.append(env_values)
            all_individual_obs.append(env_individual_obs)
        
        return all_actions, all_log_probs, all_values, all_individual_obs
    
    def collect_rollout(self, n_steps=128):
        """收集经验"""
        self.reset_buffers()
        
        # 重置环境
        obs_list = []
        for i, env in enumerate(self.envs):
            obs = env.reset()
            obs_list.append(obs)
            self.reward_shapers[i].reset()
        
        total_goal_touches = 0
        step_rewards = []
        
        for step in range(n_steps):
            # 获取动作
            all_actions, all_log_probs, all_values, all_individual_obs = self.get_parallel_actions(obs_list)
            
            # 执行动作
            next_obs_list = []
            rewards_list = []
            dones_list = []
            
            for env_idx, env in enumerate(self.envs):
                next_obs, rewards, done, info = env.step(all_actions[env_idx])
                
                # 稳定的奖励塑形
                agent_positions = [env.agent_pos[i] for i in range(self.n_agents)]
                shaped_rewards = self.reward_shapers[env_idx].shape_rewards(
                    env, agent_positions, rewards, all_actions[env_idx])
                
                # 统计目标触碰
                for r in shaped_rewards:
                    if r >= 9.5:  # 触碰目标的奖励
                        total_goal_touches += 1
                
                next_obs_list.append(next_obs)
                rewards_list.append(shaped_rewards)
                dones_list.append(done)
                
                step_rewards.extend(shaped_rewards)
            
            # 存储经验
            for env_idx in range(self.n_parallel_envs):
                for agent_idx in range(self.n_agents):
                    self.buffers[agent_idx]['observations'].append(all_individual_obs[env_idx][agent_idx])
                    self.buffers[agent_idx]['actions'].append(all_actions[env_idx][agent_idx])
                    self.buffers[agent_idx]['log_probs'].append(all_log_probs[env_idx][agent_idx])
                    self.buffers[agent_idx]['values'].append(all_values[env_idx][agent_idx])
                    self.buffers[agent_idx]['rewards'].append(rewards_list[env_idx][agent_idx])
                    self.buffers[agent_idx]['dones'].append(dones_list[env_idx])
            
            # 重置完成的环境
            for env_idx, done in enumerate(dones_list):
                if done:
                    obs_list[env_idx] = self.envs[env_idx].reset()
                    self.reward_shapers[env_idx].reset()
                else:
                    obs_list[env_idx] = next_obs_list[env_idx]
        
        # 调试输出
        avg_step_reward = np.mean(step_rewards) if step_rewards else 0
        print(f"Rollout完成: 平均步奖励={avg_step_reward:.3f}, 目标触碰={total_goal_touches}")
        
        return self.buffers, total_goal_touches
    
    def compute_gae(self, rewards, values, dones, gamma=0.99, gae_lambda=0.95):
        """计算GAE"""
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
    
    def update_agent(self, agent_id, n_epochs=4, batch_size=64, clip_coef=0.2):
        """更新单个智能体 - 添加梯度监控"""
        buffer = self.buffers[agent_id]
        agent = self.agents[agent_id]
        optimizer = self.optimizers[agent_id]
        
        if len(buffer['rewards']) == 0:
            return {}
        
        # 计算优势
        advantages, returns = self.compute_gae(
            buffer['rewards'], buffer['values'], buffer['dones'])
        
        # 转换为张量
        observations = buffer['observations']
        actions = torch.LongTensor(buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(buffer['log_probs']).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # 标准化优势
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # PPO更新
        total_loss = 0
        total_grad_norm = 0
        n_updates = 0
        
        for epoch in range(n_epochs):
            indices = torch.randperm(len(observations))
            
            for start in range(0, len(observations), batch_size):
                end = min(start + batch_size, len(observations))
                batch_indices = indices[start:end]
                
                # 准备批次
                batch_images = torch.stack([observations[i]['image'] for i in batch_indices])
                batch_directions = torch.stack([observations[i]['direction'] for i in batch_indices])
                batch_directions = batch_directions.squeeze(-1)
                
                batch_obs = {
                    'image': batch_images,
                    'direction': batch_directions
                }
                
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # 前向传播
                _, new_log_probs, entropy, new_values = agent.get_action_and_value(batch_obs, batch_actions)
                
                # 计算损失
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = 0.5 * (new_values - batch_returns).pow(2).mean()
                entropy_loss = -self.entropy_coef * entropy.mean()
                
                total_loss_batch = policy_loss + value_loss + entropy_loss
                
                # 更新
                optimizer.zero_grad()
                total_loss_batch.backward()
                
                # 计算梯度范数
                grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                total_grad_norm += grad_norm.item()
                n_updates += 1
        
        avg_grad_norm = total_grad_norm / max(1, n_updates)
        self.gradient_norms.append(avg_grad_norm)
        
        # 检测梯度异常
        if avg_grad_norm > 10.0:
            print(f"⚠️ 智能体{agent_id}梯度范数异常: {avg_grad_norm:.2f}")
        
        return {
            'loss': total_loss / max(1, n_updates),
            'grad_norm': avg_grad_norm
        }
    
    def update_all_agents(self):
        """更新所有智能体"""
        losses = []
        grad_norms = []
        
        for i in range(self.n_agents):
            metrics = self.update_agent(i)
            if 'loss' in metrics:
                losses.append(metrics['loss'])
                grad_norms.append(metrics['grad_norm'])
        
        return {
            'avg_loss': np.mean(losses) if losses else 0,
            'avg_grad_norm': np.mean(grad_norms) if grad_norms else 0
        }
    
    def check_training_health(self, current_reward, episode):
        """检查训练健康状况"""
        self.performance_history.append(current_reward)
        
        if len(self.performance_history) < 100:
            return True
        
        # 检查是否出现严重下降
        recent_avg = np.mean(self.performance_history[-50:])
        older_avg = np.mean(self.performance_history[-100:-50])
        
        if recent_avg < older_avg - 100:  # 下降超过100
            print(f"🚨 警告：训练可能出现问题！")
            print(f"   最近50轮平均: {recent_avg:.2f}")
            print(f"   之前50轮平均: {older_avg:.2f}")
            print(f"   下降幅度: {older_avg - recent_avg:.2f}")
            
            # 检查梯度
            if len(self.gradient_norms) > 0:
                recent_grad = np.mean(self.gradient_norms[-10:])
                print(f"   最近梯度范数: {recent_grad:.4f}")
            
            return False
        
        return True
    
    def save_models(self, path_prefix):
        """保存模型"""
        if "models6" not in path_prefix:
            path_prefix = path_prefix.replace("models/", "models6/").replace("model2/", "models6/")
        
        os.makedirs(os.path.dirname(path_prefix) if os.path.dirname(path_prefix) else "models6", exist_ok=True)
        for i in range(self.n_agents):
            path = f"{path_prefix}_agent_{i}.pth"
            torch.save(self.agents[i].state_dict(), path)
        print(f"✅ 已保存模型到: {path_prefix}_agent_*.pth")
    
    def load_models(self, path_prefix):
        """加载模型"""
        if "models6" not in path_prefix:
            path_prefix = path_prefix.replace("models/", "models6/").replace("model2/", "models6/")
            
        for i in range(self.n_agents):
            path = f"{path_prefix}_agent_{i}.pth"
            if os.path.exists(path):
                self.agents[i].load_state_dict(torch.load(path, map_location=self.device))
            else:
                print(f"Warning: 模型文件 {path} 不存在")
        print(f"✅ 已加载模型: {path_prefix}_agent_*.pth")
    
    def close_envs(self):
        """关闭环境"""
        for env in self.envs:
            env.close()


def train_stable_ppo(
    env_name="MultiGrid-Cluttered-Fixed-15x15",
    n_episodes=50000,  # 符合文档要求
    n_steps=128,
    n_parallel_envs=4,
    use_wandb=True,
    project_name="stable-goal-focused-ppo",
    experiment_name=None
):
    """稳定版PPO训练"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🛡️ 启动稳定版目标导向多智能体PPO训练")
    print("="*60)
    print("🔧 稳定性改进:")
    print("   ✅ 学习率降低: 3e-4 → 1e-4")
    print("   ✅ 奖励设计优化，避免负值累积")
    print("   ✅ 添加梯度监控和异常检测")
    print("   ✅ 更保守的奖励塑形")
    print("   ✅ 早停机制防止训练崩溃")
    print("="*60)
    
    # 获取环境信息
    temp_env = gym.make(env_name)
    n_agents = temp_env.n_agents
    temp_env.close()
    
    print(f"🎯 环境: {env_name}")
    print(f"🤖 智能体数量: {n_agents}")
    print(f"📈 训练轮数: {n_episodes}")
    print(f"🔄 并行环境: {n_parallel_envs}")
    print(f"📱 设备: {device}")
    
    # 实验名称
    if experiment_name is None:
        experiment_name = f"stable_{env_name}_{n_agents}agents_{int(time.time())}"
    
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
                    "n_steps": n_steps,
                    "n_parallel_envs": n_parallel_envs,
                    "device": str(device),
                    "algorithm": "Stable Goal-Focused Independent PPO",
                    "learning_rate": 1e-4,
                    "goal_reward": 10.0,
                    "distance_reward_scale": 0.2,
                    "max_stationary_penalty": -0.05
                },
                tags=["stable", "goal-focused", "multi-agent", "ppo", "multigrid"]
            )
            print("✅ WandB日志已初始化")
        except Exception as e:
            print(f"⚠️ WandB初始化失败: {e}")
            use_wandb = False
    else:
        use_wandb = False
        print("📝 使用本地日志")
    
    # 创建稳定控制器
    controller = StablePPOController(
        env_name, n_agents, device, 
        lr=1e-4,  # 降低学习率
        n_parallel_envs=n_parallel_envs
    )
    
    # 训练指标
    episode_rewards = []
    collective_rewards = []
    best_collective_reward = float('-inf')
    total_goal_touches = 0
    
    print(f"\n🛡️ 开始稳定版训练...")
    start_time = time.time()
    
    for episode in range(n_episodes):
        episode_start = time.time()
        
        # 收集经验
        buffers, rollout_goal_touches = controller.collect_rollout(n_steps)
        total_goal_touches += rollout_goal_touches
        
        # 更新智能体
        update_metrics = controller.update_all_agents()
        
        # 计算奖励
        agent_rewards = []
        for i in range(n_agents):
            agent_reward = sum(buffers[i]['rewards'])
            agent_rewards.append(agent_reward)
        
        collective_reward = sum(agent_rewards)
        episode_rewards.append(agent_rewards)
        collective_rewards.append(collective_reward)
        
        # 检查训练健康状况
        training_healthy = controller.check_training_health(collective_reward, episode)
        if not training_healthy and episode > 1000:
            print("🚨 检测到训练异常，建议停止并检查参数！")
            break
        
        # 保存最佳模型
        if collective_reward > best_collective_reward:
            best_collective_reward = collective_reward
            controller.save_models(f"models6/best_{experiment_name}")
        
        # 计算平均指标
        episode_time = time.time() - episode_start
        
        if len(collective_rewards) >= 100:
            avg_collective = np.mean(collective_rewards[-100:])
            avg_individual = np.mean([np.mean([ep[i] for ep in episode_rewards[-100:]]) for i in range(n_agents)])
        else:
            avg_collective = collective_reward
            avg_individual = np.mean(agent_rewards)
        
        # WandB日志
        if use_wandb:
            log_dict = {
                "episode": episode,
                "collective_reward": collective_reward,
                "avg_collective_reward_100": avg_collective,
                "avg_individual_reward_100": avg_individual,
                "best_collective_reward": best_collective_reward,
                "episode_time": episode_time,
                "total_samples": (episode + 1) * n_steps * n_parallel_envs,
                "goal_touches_total": total_goal_touches,
                "goal_touches_episode": rollout_goal_touches,
                "avg_loss": update_metrics.get('avg_loss', 0),
                "avg_grad_norm": update_metrics.get('avg_grad_norm', 0),
                "training_healthy": training_healthy
            }
            
            # 添加个体智能体奖励
            for i in range(n_agents):
                log_dict[f"agent_{i}_reward"] = agent_rewards[i]
            
            wandb.log(log_dict)
        
        # 控制台输出 - 增加调试信息
        if episode % 50 == 0:
            total_time = time.time() - start_time
            eps_per_hour = episode * 3600 / total_time if total_time > 0 else 0
            
            print(f"Episode {episode:6d} | "
                  f"集体奖励: {avg_collective:8.2f} | "
                  f"个体平均: {avg_individual:8.2f} | "
                  f"最佳: {best_collective_reward:8.2f} | "
                  f"触碰目标: {rollout_goal_touches:2d} | "
                  f"总触碰: {total_goal_touches:4d} | "
                  f"梯度: {update_metrics.get('avg_grad_norm', 0):.3f} | "
                  f"速度: {eps_per_hour:.1f} ep/h")
            
            if not training_healthy:
                print("        🚨 训练健康状况异常！")
        
        # 定期保存
        if episode % 10000 == 0 and episode > 0:
            controller.save_models(f"models6/{experiment_name}_ep{episode}")
    
    # 最终保存
    controller.save_models(f"models6/{experiment_name}_final")
    controller.close_envs()
    
    # 最终结果
    total_time = time.time() - start_time
    
    print(f"\n🎉 稳定版训练完成！")
    print("="*60)
    print(f"⏱️ 总时间: {total_time/3600:.2f} 小时")
    print(f"🏆 最佳集体奖励: {best_collective_reward:.2f}")
    print(f"📈 最终100轮平均: {np.mean(collective_rewards[-100:]):.2f}")
    print(f"🎯 总触碰目标次数: {total_goal_touches}")
    print(f"📊 平均每轮触碰: {total_goal_touches/len(episode_rewards):.2f}")
    
    # 保存结果
    results = {
        'episode_rewards': episode_rewards,
        'collective_rewards': collective_rewards,
        'best_collective_reward': best_collective_reward,
        'total_goal_touches': total_goal_touches,
        'total_time_hours': total_time / 3600,
        'config': {
            'env_name': env_name,
            'n_agents': n_agents,
            'n_episodes': len(episode_rewards),
            'experiment_name': experiment_name,
            'learning_rate': 1e-4,
            'stable_features': [
                'reduced_learning_rate',
                'stable_reward_shaping',
                'gradient_monitoring',
                'training_health_check'
            ]
        }
    }
    
    os.makedirs('models6', exist_ok=True)
    with open(f"models6/results_{experiment_name}.json", 'w') as f:
        json.dump(results, f)
    
    if use_wandb:
        wandb.finish()
    
    return controller, episode_rewards, collective_rewards, experiment_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v4稳定版目标导向多智能体PPO")
    parser.add_argument("--env", type=str, default="MultiGrid-Cluttered-Fixed-15x15", 
                        help="环境名称")
    parser.add_argument("--episodes", type=int, default=50000, 
                        help="训练轮数")
    parser.add_argument("--parallel-envs", type=int, default=4, 
                        help="并行环境数量")
    parser.add_argument("--steps", type=int, default=128, 
                        help="每次rollout的步数")
    parser.add_argument("--no-wandb", action="store_true", 
                        help="禁用WandB日志")
    parser.add_argument("--project", type=str, default="stable-goal-focused-ppo", 
                        help="WandB项目名称")
    parser.add_argument("--name", type=str, default=None, 
                        help="实验名称")
    
    args = parser.parse_args()
    
    print("🛡️ 启动稳定版目标导向多智能体PPO训练...")
    print("🔧 主要修复:")
    print("   ✅ 学习率降低防止参数更新过大")
    print("   ✅ 奖励设计避免负值累积")
    print("   ✅ 梯度监控防止训练崩溃")
    print("   ✅ 早停机制保护训练过程")
    print("="*70)
    
    controller, episode_rewards, collective_rewards, experiment_name = train_stable_ppo(
        env_name=args.env,
        n_episodes=args.episodes,
        n_steps=args.steps,
        n_parallel_envs=args.parallel_envs,
        use_wandb=not args.no_wandb,
        project_name=args.project,
        experiment_name=args.name
    )
    
    print(f"\n✅ 稳定版训练完成!")
    print(f"📁 模型已保存到: models6/{experiment_name}_final")
    print(f"📊 结果已保存到: models6/results_{experiment_name}.json")