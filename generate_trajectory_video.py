"""
生成轨迹视频帧脚本 - 基于v8训练模型
"""

import gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from torch.distributions.categorical import Categorical

from envs import gym_multigrid
from envs.gym_multigrid import multigrid_envs


class SimplePPOAgent(nn.Module):
    """与v8相同的网络结构"""
    
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


class VideoGenerator:
    """视频帧生成器"""
    
    def __init__(self, model_path_prefix, env_name="MultiGrid-Cluttered-Fixed-15x15"):
        self.env_name = env_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建环境
        self.env = gym.make(env_name)
        self.n_agents = self.env.n_agents
        
        # 加载模型
        self.agents = []
        for i in range(self.n_agents):
            agent = SimplePPOAgent().to(self.device)
            agent.load_state_dict(torch.load(f"{model_path_prefix}_agent_{i}.pth", map_location=self.device))
            agent.eval()
            self.agents.append(agent)
        
        # 动作名称映射
        self.action_names = {
            0: "left", 1: "right", 2: "forward", 
            3: "pickup", 4: "drop", 5: "toggle", 6: "done"
        }
        
        print(f"加载完成: {self.n_agents}个智能体, 环境: {env_name}")
    
    def get_actions(self, obs):
        """获取动作"""
        actions = []
        
        for i in range(self.n_agents):
            agent_obs = {
                'image': torch.FloatTensor(obs['image'][i]).to(self.device),
                'direction': torch.LongTensor([obs['direction'][i]]).to(self.device)
            }
            
            with torch.no_grad():
                action, _, _, _ = self.agents[i].get_action_and_value(agent_obs)
            
            actions.append(action.item())
        
        return actions
    
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
        """v8风格的奖励塑形"""
        shaped_rewards = []
        goal_pos = self.get_goal_position()
        
        for i in range(self.n_agents):
            pos = np.array(agent_positions[i])
            action = actions[i]
            
            # 触碰目标 - 大奖励
            if original_rewards[i] > 0:
                shaped_rewards.append(5.0)
                continue
            
            reward = 0.0
            
            # 距离奖励
            current_dist = np.linalg.norm(pos - goal_pos)
            if hasattr(self, 'prev_distances') and self.prev_distances[i] is not None:
                dist_change = self.prev_distances[i] - current_dist
                reward += dist_change * 0.2
            
            # 移动奖励
            if hasattr(self, 'prev_positions') and self.prev_positions[i] is not None:
                if not np.array_equal(pos, self.prev_positions[i]):
                    reward += 0.02
                else:
                    reward -= 0.02
            
            # 行动奖励
            if action == 2:  # forward
                reward += 0.02
            elif action in [0, 1]:  # turn
                reward += 0.01
                
            reward = max(reward, -0.2)
            shaped_rewards.append(reward)
            
        return shaped_rewards
    
    def generate_trajectory(self, max_steps=100):
        """生成轨迹数据"""
        obs = self.env.reset()
        
        # 初始化记录
        trajectory_data = []
        episode_rewards = [0] * self.n_agents
        collective_rewards = [0]
        
        # 初始化奖励塑形状态
        self.prev_distances = [None] * self.n_agents
        self.prev_positions = [None] * self.n_agents
        
        for step in range(max_steps):
            # 获取动作
            actions = self.get_actions(obs)
            
            # 执行动作
            next_obs, original_rewards, done, info = self.env.step(actions)
            
            # 计算塑形奖励
            agent_positions = [self.env.agent_pos[i] for i in range(self.n_agents)]
            shaped_rewards = self.shape_rewards(agent_positions, original_rewards, actions)
            
            # 更新累积奖励
            for i in range(self.n_agents):
                episode_rewards[i] += shaped_rewards[i]
            collective_rewards.append(sum(episode_rewards))
            
            # 获取智能体局部观察图像
            agents_partial_images = []
            for i in range(self.n_agents):
                partial_img = self.env.get_obs_render(obs['image'][i])
                agents_partial_images.append(partial_img)
            
            # 记录当前帧数据
            frame_data = {
                'step': step,
                'full_image': self.env.render('rgb_array'),
                'agents_partial_images': agents_partial_images,
                'actions': actions,
                'shaped_rewards': shaped_rewards,
                'episode_rewards': episode_rewards.copy(),
                'collective_rewards': collective_rewards.copy(),
                'done': done
            }
            trajectory_data.append(frame_data)
            
            # 更新状态
            obs = next_obs
            
            # 更新奖励塑形状态
            goal_pos = self.get_goal_position()
            for i in range(self.n_agents):
                pos = np.array(agent_positions[i])
                self.prev_distances[i] = np.linalg.norm(pos - goal_pos)
                self.prev_positions[i] = pos.copy()
            
            if done:
                break
        
        return trajectory_data
    
    def plot_frame(self, frame_data, output_path):
        """绘制单帧图像"""
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 2+self.n_agents, figure=fig, hspace=0.3, wspace=0.3)
        
        step = frame_data['step']
        full_image = frame_data['full_image']
        agents_partial_images = frame_data['agents_partial_images']
        actions = frame_data['actions']
        shaped_rewards = frame_data['shaped_rewards']
        episode_rewards = frame_data['episode_rewards']
        collective_rewards = frame_data['collective_rewards']
        
        # 1. 完整环境状态 (左上角 2x2)
        ax_full = fig.add_subplot(gs[:2, :2])
        ax_full.imshow(full_image)
        ax_full.set_title('Full Environment State', fontsize=14, fontweight='bold')
        ax_full.axis('off')
        
        # 2. 智能体局部观察 (右上角)
        for i in range(self.n_agents):
            ax_partial = fig.add_subplot(gs[0, 2+i])
            ax_partial.imshow(agents_partial_images[i])
            ax_partial.set_title(f'Agent{i} Partial Obs', fontsize=10)
            ax_partial.axis('off')
        
        # 3. 集体累积奖励 (左下角)
        ax_collective = fig.add_subplot(gs[2, :2])
        steps = list(range(len(collective_rewards)))
        ax_collective.plot(steps, collective_rewards, 'b-', linewidth=2)
        if step > 0:
            ax_collective.plot(step, collective_rewards[step], 'ro', markersize=8)
        ax_collective.set_title('Collective Return', fontsize=12, fontweight='bold')
        ax_collective.set_xlabel('Step')
        ax_collective.set_ylabel('Collective Return')
        ax_collective.grid(True, alpha=0.3)
        
        # 4. 各智能体累积奖励和动作信息 (右下角)
        for i in range(self.n_agents):
            ax_agent = fig.add_subplot(gs[1, 2+i])
            
            # 绘制累积奖励曲线
            agent_cumulative = [0]
            cumsum = 0
            for j in range(step+1):
                if j < len(collective_rewards)-1:
                    # 这里简化处理，假设平均分配
                    cumsum += collective_rewards[j+1] - collective_rewards[j]
                agent_cumulative.append(cumsum / self.n_agents)
            
            ax_agent.plot(range(len(agent_cumulative)), agent_cumulative, 'g-', linewidth=2)
            if step > 0:
                ax_agent.plot(step, agent_cumulative[step], 'ro', markersize=6)
            
            ax_agent.set_title(f'Agent{i} Return', fontsize=10)
            ax_agent.set_xlabel('Step')
            ax_agent.set_ylabel('Return')
            ax_agent.grid(True, alpha=0.3)
            
            # 添加动作和奖励文本信息
            action_name = self.action_names.get(actions[i], f"action_{actions[i]}")
            reward_text = f"a^{i}_t={step}: {action_name}\nR_t={step}: {shaped_rewards[i]:.3f}"
            
            # 在图的右下角添加文本
            ax_agent.text(0.98, 0.02, reward_text, transform=ax_agent.transAxes,
                         fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        # 添加整体标题
        fig.suptitle(f'Multi-Agent Trajectory - Step {step}', fontsize=16, fontweight='bold')
        
        # 保存图像
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_video_frames(self, model_path_prefix, output_dir="trajectory_frames", max_steps=100):
        """生成视频帧"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"生成轨迹数据...")
        trajectory_data = self.generate_trajectory(max_steps)
        
        print(f"生成 {len(trajectory_data)} 帧图像...")
        for i, frame_data in enumerate(trajectory_data):
            output_path = os.path.join(output_dir, f"frame_{i:05d}.png")
            self.plot_frame(frame_data, output_path)
            
            if i % 10 == 0:
                print(f"已生成 {i+1}/{len(trajectory_data)} 帧")
        
        print(f"完成! 图像保存在: {output_dir}")
        return len(trajectory_data)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="生成轨迹视频帧")
    parser.add_argument("--model-path", type=str, required=True, 
                       help="模型路径前缀，例如: models8/best_performance")
    parser.add_argument("--env", type=str, default="MultiGrid-Cluttered-Fixed-15x15",
                       help="环境名称")
    parser.add_argument("--output-dir", type=str, default="trajectory_frames",
                       help="输出目录")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="最大步数")
    
    args = parser.parse_args()
    
    # 创建视频生成器
    generator = VideoGenerator(args.model_path, args.env)
    
    # 生成帧
    num_frames = generator.generate_video_frames(
        args.model_path, 
        args.output_dir, 
        args.max_steps
    )
    
    print(f"\n生成完成!")
    print(f"总帧数: {num_frames}")
    print(f"输出目录: {args.output_dir}")
    
    # 生成视频的命令提示
    print(f"\n生成视频命令:")
    print(f"ffmpeg -r 10 -i {args.output_dir}/frame_%05d.png -c:v libx264 -pix_fmt yuv420p trajectory_video.mp4")


if __name__ == "__main__":
    main()