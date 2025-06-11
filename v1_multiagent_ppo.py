"""
ulti-Agent PPO Implementation for Social RL Take-home Interview

Features:
- Independent PPO (IPPO) with multiple agents
- WandB logging for learning curves
- Trajectory visualization
- Model saving/loading
- Complete training pipeline for MultiGrid-Cluttered-Fixed-15x15
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
from collections import defaultdict
import time
import json

# Import needed to trigger env registration
from envs import gym_multigrid
from envs.gym_multigrid import multigrid_envs

# Fix wandb numpy compatibility
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Try to import wandb with fallback
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging will be disabled")


class MultiGridPPOAgent(nn.Module):
    """Individual PPO Agent - State-of-the-art architecture for MultiGrid"""
    
    def __init__(self, n_actions=7):
        super().__init__()
        
        # Image processing network - optimized for 5x5 grids
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Direction embedding (4 directions)
        self.direction_embed = nn.Embedding(4, 16)
        
        # Shared feature network
        self.shared = nn.Sequential(
            nn.Linear(64 + 16, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # Actor and Critic heads
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, 0.5)
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.orthogonal_(module.weight, 1.0)
            
    def forward(self, obs):
        """Forward pass"""
        image = obs['image']
        direction = obs['direction']
        
        # Convert image format: (B, H, W, C) -> (B, C, H, W)
        if len(image.shape) == 4:
            image = image.permute(0, 3, 1, 2)
        elif len(image.shape) == 3:
            image = image.permute(2, 0, 1).unsqueeze(0)
        
        # Ensure direction has correct shape
        if len(direction.shape) > 1:
            direction = direction.squeeze(-1)
        if len(direction.shape) == 0:
            direction = direction.unsqueeze(0)
        
        # Feature extraction
        image_features = self.image_conv(image.float())
        direction_features = self.direction_embed(direction.long())
        
        # Ensure batch dimensions match
        batch_size = image_features.shape[0]
        if direction_features.shape[0] != batch_size:
            if direction_features.shape[0] == 1:
                direction_features = direction_features.repeat(batch_size, 1)
        
        # Combine features
        combined = torch.cat([image_features, direction_features], dim=-1)
        shared_features = self.shared(combined)
        
        # Output
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        
        return action_logits, value.squeeze(-1)
    
    def get_action_and_value(self, obs, action=None):
        """Get action and value with proper distributions"""
        action_logits, value = self.forward(obs)
        dist = Categorical(logits=action_logits)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value


class MultiAgentPPOController:
    """
    Multi-Agent PPO Controller implementing Independent PPO (IPPO)
    
    This is the main contribution - a metacontroller that manages multiple
    independent PPO agents in a multi-agent environment.
    """
    
    def __init__(self, env_name, n_agents, device, lr=3e-4, n_parallel_envs=4):
        self.env_name = env_name
        self.n_agents = n_agents
        self.device = device
        self.n_parallel_envs = n_parallel_envs
        
        # Create independent PPO agents
        self.agents = []
        self.optimizers = []
        
        for i in range(n_agents):
            agent = MultiGridPPOAgent(n_actions=7).to(device)
            optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
            
            self.agents.append(agent)
            self.optimizers.append(optimizer)
        
        # Create parallel environments for faster data collection
        self.envs = self._create_parallel_envs()
        
        # Experience buffers - separate for each agent
        self.reset_buffers()
        
        print(f"✅ Created {n_agents} independent PPO agents")
        print(f"✅ Using {n_parallel_envs} parallel environments")
        print(f"✅ Device: {device}")
    
    def _create_parallel_envs(self):
        """Create parallel environments for faster training"""
        envs = []
        for i in range(self.n_parallel_envs):
            env = gym.make(self.env_name)
            envs.append(env)
        return envs
    
    def reset_buffers(self):
        """Reset experience buffers for all agents"""
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
    
    def preprocess_parallel_obs(self, obs_list):
        """Preprocess observations from parallel environments"""
        agent_observations = [[] for _ in range(self.n_agents)]
        
        for env_obs in obs_list:
            for i in range(self.n_agents):
                agent_obs = {
                    'image': torch.FloatTensor(env_obs['image'][i]).to(self.device),
                    'direction': torch.LongTensor([env_obs['direction'][i]]).to(self.device)
                }
                agent_observations[i].append(agent_obs)
        
        # Batch observations for each agent
        batched_agent_obs = []
        for i in range(self.n_agents):
            if len(agent_observations[i]) > 0:
                batch_images = torch.stack([obs['image'] for obs in agent_observations[i]])
                batch_directions = torch.stack([obs['direction'] for obs in agent_observations[i]])
                batch_directions = batch_directions.squeeze(-1)
                
                batched_obs = {
                    'image': batch_images,
                    'direction': batch_directions
                }
                batched_agent_obs.append(batched_obs)
            else:
                batched_agent_obs.append(None)
        
        return batched_agent_obs, agent_observations
    
    def get_actions(self, obs):
        """Get actions from all agents (single environment version)"""
        agent_observations = []
        
        for i in range(self.n_agents):
            agent_obs = {
                'image': torch.FloatTensor(obs['image'][i]).to(self.device),
                'direction': torch.LongTensor([obs['direction'][i]]).to(self.device)
            }
            agent_observations.append(agent_obs)
        
        actions = []
        log_probs = []
        values = []
        entropies = []
        
        for i in range(self.n_agents):
            with torch.no_grad():
                action, log_prob, entropy, value = self.agents[i].get_action_and_value(agent_observations[i])
            
            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value.item())
            entropies.append(entropy.item())
        
        return actions, log_probs, values, entropies, agent_observations
        """Get actions from all agents in parallel environments"""
        batched_agent_obs, individual_obs = self.preprocess_parallel_obs(obs_list)
        
        all_actions = []
        all_log_probs = []
        all_values = []
        all_entropies = []
        
        for i in range(self.n_agents):
            if batched_agent_obs[i] is not None:
                with torch.no_grad():
                    actions, log_probs, entropies, values = self.agents[i].get_action_and_value(batched_agent_obs[i])
                
                all_actions.append(actions.cpu().numpy())
                all_log_probs.append(log_probs.cpu().numpy())
                all_values.append(values.cpu().numpy())
                all_entropies.append(entropies.cpu().numpy())
            else:
                all_actions.append(np.array([]))
                all_log_probs.append(np.array([]))
                all_values.append(np.array([]))
                all_entropies.append(np.array([]))
        
        # Rearrange to per-environment action lists
        env_actions = []
        for env_idx in range(self.n_parallel_envs):
            env_action = []
            for agent_idx in range(self.n_agents):
                if len(all_actions[agent_idx]) > env_idx:
                    env_action.append(all_actions[agent_idx][env_idx])
                else:
                    env_action.append(0)  # Default action
            env_actions.append(env_action)
        
        return env_actions, all_log_probs, all_values, all_entropies, individual_obs
    
    def collect_parallel_rollout(self, n_steps=128):
        """Collect experience using parallel environments"""
        self.reset_buffers()
        
        # Reset all environments
        obs_list = []
        for env in self.envs:
            obs = env.reset()
            obs_list.append(obs)
        
        for step in range(n_steps):
            # Get actions from all agents in all environments
            env_actions, log_probs_list, values_list, entropies_list, individual_obs = self.get_parallel_actions(obs_list)
            
            # Execute actions in all environments
            next_obs_list = []
            rewards_list = []
            dones_list = []
            
            for env_idx, env in enumerate(self.envs):
                next_obs, rewards, done, info = env.step(env_actions[env_idx])
                next_obs_list.append(next_obs)
                rewards_list.append(rewards)
                dones_list.append(done)
            
            # Store experiences in agent buffers
            for env_idx in range(self.n_parallel_envs):
                for agent_idx in range(self.n_agents):
                    if len(individual_obs[agent_idx]) > env_idx:
                        self.buffers[agent_idx]['observations'].append(individual_obs[agent_idx][env_idx])
                        self.buffers[agent_idx]['actions'].append(env_actions[env_idx][agent_idx])
                        
                        if len(log_probs_list[agent_idx]) > env_idx:
                            self.buffers[agent_idx]['log_probs'].append(log_probs_list[agent_idx][env_idx])
                            self.buffers[agent_idx]['values'].append(values_list[agent_idx][env_idx])
                        else:
                            self.buffers[agent_idx]['log_probs'].append(0.0)
                            self.buffers[agent_idx]['values'].append(0.0)
                        
                        self.buffers[agent_idx]['rewards'].append(rewards_list[env_idx][agent_idx])
                        self.buffers[agent_idx]['dones'].append(dones_list[env_idx])
            
            # Reset completed environments
            for env_idx, done in enumerate(dones_list):
                if done:
                    obs_list[env_idx] = self.envs[env_idx].reset()
                else:
                    obs_list[env_idx] = next_obs_list[env_idx]
        
        return self.buffers
    
    def compute_gae(self, rewards, values, dones, gamma=0.99, gae_lambda=0.95):
        """Compute Generalized Advantage Estimation"""
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
    
    def update_agent(self, agent_id, n_epochs=4, batch_size=128, clip_coef=0.2):
        """Update individual agent using PPO"""
        buffer = self.buffers[agent_id]
        agent = self.agents[agent_id]
        optimizer = self.optimizers[agent_id]
        
        if len(buffer['rewards']) == 0:
            return {}
        
        # Compute advantages
        advantages, returns = self.compute_gae(
            buffer['rewards'], buffer['values'], buffer['dones'])
        
        # Convert to tensors
        observations = buffer['observations']
        actions = torch.LongTensor(buffer['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(buffer['log_probs']).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # PPO updates
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for epoch in range(n_epochs):
            indices = torch.randperm(len(observations))
            
            for start in range(0, len(observations), batch_size):
                end = min(start + batch_size, len(observations))
                batch_indices = indices[start:end]
                
                # Prepare batch
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
                
                # Forward pass
                _, new_log_probs, entropy, new_values = agent.get_action_and_value(batch_obs, batch_actions)
                
                # Compute losses
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = 0.5 * (new_values - batch_returns).pow(2).mean()
                entropy_loss = -0.01 * entropy.mean()
                
                total_loss = policy_loss + value_loss + entropy_loss
                
                # Update
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        return {
            'policy_loss': total_policy_loss / (n_epochs * len(range(0, len(observations), batch_size))),
            'value_loss': total_value_loss / (n_epochs * len(range(0, len(observations), batch_size))),
            'entropy_loss': total_entropy_loss / (n_epochs * len(range(0, len(observations), batch_size)))
        }
    
    def update_all_agents(self):
        """Update all agents and return aggregated metrics"""
        metrics = {}
        for i in range(self.n_agents):
            agent_metrics = self.update_agent(i)
            for key, value in agent_metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
        
        # Average metrics across agents
        avg_metrics = {}
        for key, values in metrics.items():
            if values:
                avg_metrics[f'avg_{key}'] = np.mean(values)
        
        return avg_metrics
    
    def save_models(self, path_prefix):
        """Save all agent models"""
        os.makedirs(os.path.dirname(path_prefix) if os.path.dirname(path_prefix) else ".", exist_ok=True)
        for i in range(self.n_agents):
            path = f"{path_prefix}_agent_{i}.pth"
            torch.save(self.agents[i].state_dict(), path)
        print(f"✅ Saved {self.n_agents} agent models to: {path_prefix}_agent_*.pth")
    
    def load_models(self, path_prefix):
        """Load all agent models"""
        for i in range(self.n_agents):
            path = f"{path_prefix}_agent_{i}.pth"
            if os.path.exists(path):
                self.agents[i].load_state_dict(torch.load(path, map_location=self.device))
            else:
                print(f"Warning: Model file {path} not found")
        print(f"✅ Loaded {self.n_agents} agent models from: {path_prefix}_agent_*.pth")
    
    def close_envs(self):
        """Close all environments"""
        for env in self.envs:
            env.close()


def train_multiagent_ppo(
    env_name, 
    n_episodes=100000,  # Full training episodes as required
    n_steps=128, 
    n_parallel_envs=4,
    use_wandb=True,
    project_name="multigrid-multiagent-ppo",
    experiment_name=None
):
    """
    Train Multi-Agent PPO with full logging and visualization
    
    Args:
        env_name: MultiGrid environment name
        n_episodes: Number of training episodes (100000 for full training)
        n_steps: Steps per rollout
        n_parallel_envs: Number of parallel environments
        use_wandb: Whether to use Weights & Biases logging
        project_name: WandB project name
        experiment_name: Custom experiment name
    """
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Multi-Agent PPO Training")
    print(f"📱 Device: {device}")
    
    # Create temporary environment to get info
    temp_env = gym.make(env_name)
    n_agents = temp_env.n_agents
    temp_env.close()
    
    print(f"🎯 Environment: {env_name}")
    print(f"🤖 Number of agents: {n_agents}")
    print(f"📊 Training episodes: {n_episodes}")
    print(f"🔄 Parallel environments: {n_parallel_envs}")
    print(f"📈 Steps per rollout: {n_steps}")
    print(f"💾 Total samples per episode: {n_steps * n_parallel_envs}")
    
    # Setup experiment name
    if experiment_name is None:
        experiment_name = f"{env_name}_{n_agents}agents_{int(time.time())}"
    
    # Initialize WandB
    if use_wandb and WANDB_AVAILABLE:
        try:
            # Fix numpy compatibility
            import numpy as np
            if not hasattr(np, 'float_'):
                np.float_ = np.float64
                
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
                    "algorithm": "Independent PPO (IPPO)",
                    "framework": "PyTorch"
                },
                tags=["multi-agent", "ppo", "multigrid", "social-rl"]
            )
            print("✅ WandB logging initialized")
        except Exception as e:
            print(f"⚠️  WandB initialization failed: {e}")
            use_wandb = False
    else:
        use_wandb = False
        print("📝 Using local logging only")
    
    # Create controller
    controller = MultiAgentPPOController(env_name, n_agents, device, n_parallel_envs=n_parallel_envs)
    
    # Training metrics
    episode_rewards = []
    collective_rewards = []
    training_metrics = defaultdict(list)
    
    print("\n🎯 Starting training loop...")
    start_time = time.time()
    best_collective_reward = float('-inf')
    
    for episode in range(n_episodes):
        episode_start = time.time()
        
        # Collect experience
        buffers = controller.collect_parallel_rollout(n_steps)
        
        # Update all agents
        update_metrics = controller.update_all_agents()
        
        # Calculate rewards
        agent_rewards = []
        for i in range(n_agents):
            agent_reward = sum(buffers[i]['rewards'])
            agent_rewards.append(agent_reward)
        
        collective_reward = sum(agent_rewards)
        episode_rewards.append(agent_rewards)
        collective_rewards.append(collective_reward)
        
        # Track best performance
        if collective_reward > best_collective_reward:
            best_collective_reward = collective_reward
            # Save best model
            controller.save_models(f"models/best_{experiment_name}")
        
        # Log metrics
        episode_time = time.time() - episode_start
        
        # Calculate averages
        if len(collective_rewards) >= 10:
            avg_collective = np.mean(collective_rewards[-10:])
            avg_individual = np.mean([np.mean([ep[i] for ep in episode_rewards[-10:]]) for i in range(n_agents)])
        else:
            avg_collective = collective_reward
            avg_individual = np.mean(agent_rewards)
        
        # WandB logging
        if use_wandb:
            log_dict = {
                "episode": episode,
                "collective_reward": collective_reward,
                "avg_collective_reward_10": avg_collective,
                "avg_individual_reward_10": avg_individual,
                "best_collective_reward": best_collective_reward,
                "episode_time": episode_time,
                "total_samples": (episode + 1) * n_steps * n_parallel_envs
            }
            
            # Add individual agent rewards
            for i in range(n_agents):
                log_dict[f"agent_{i}_reward"] = agent_rewards[i]
            
            # Add training metrics
            for key, value in update_metrics.items():
                log_dict[key] = value
            
            wandb.log(log_dict)
        
        # Console logging
        if episode % 10 == 0:
            total_time = time.time() - start_time
            eps_per_hour = episode * 3600 / total_time if total_time > 0 else 0
            
            print(f"Episode {episode:6d} | "
                  f"Collective: {avg_collective:7.2f} | "
                  f"Individual: {avg_individual:7.2f} | "
                  f"Best: {best_collective_reward:7.2f} | "
                  f"Time: {episode_time:.2f}s | "
                  f"Speed: {eps_per_hour:.1f} ep/h")
        
        # Save models periodically
        if episode % 1000 == 0 and episode > 0:
            controller.save_models(f"models/{experiment_name}_ep{episode}")
            
        # Save results periodically
        if episode % 1000 == 0 and episode > 0:
            results = {
                'episode_rewards': episode_rewards,
                'collective_rewards': collective_rewards,
                'config': {
                    'env_name': env_name,
                    'n_agents': n_agents,
                    'n_episodes': episode,
                    'experiment_name': experiment_name
                }
            }
            with open(f"results_{experiment_name}_ep{episode}.json", 'w') as f:
                json.dump(results, f)
    
    # Final save
    controller.save_models(f"models/{experiment_name}_final")
    controller.close_envs()
    
    # Final results
    total_time = time.time() - start_time
    print(f"\n🎉 Training completed!")
    print(f"⏱️  Total time: {total_time/3600:.2f} hours")
    print(f"🏆 Best collective reward: {best_collective_reward:.2f}")
    print(f"📈 Final 100-episode average: {np.mean(collective_rewards[-100:]):.2f}")
    
    # Save final results
    results = {
        'episode_rewards': episode_rewards,
        'collective_rewards': collective_rewards,
        'best_collective_reward': best_collective_reward,
        'total_time_hours': total_time / 3600,
        'config': {
            'env_name': env_name,
            'n_agents': n_agents,
            'n_episodes': n_episodes,
            'experiment_name': experiment_name
        }
    }
    
    with open(f"final_results_{experiment_name}.json", 'w') as f:
        json.dump(results, f)
    
    # Plot final results
    plt.figure(figsize=(15, 5))
    
    # Collective rewards
    plt.subplot(1, 3, 1)
    plt.plot(collective_rewards)
    plt.title('Collective Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Individual agent rewards
    plt.subplot(1, 3, 2)
    for i in range(n_agents):
        individual_rewards = [ep[i] for ep in episode_rewards]
        plt.plot(individual_rewards, label=f'Agent {i}', alpha=0.7)
    plt.title('Individual Agent Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    
    # Moving average
    plt.subplot(1, 3, 3)
    window = min(1000, len(collective_rewards) // 10)
    if len(collective_rewards) > window:
        moving_avg = np.convolve(collective_rewards, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg)
        plt.title(f'Moving Average ({window} episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
    
    plt.tight_layout()
    plt.savefig(f'training_curves_{experiment_name}.png', dpi=300, bbox_inches='tight')
    print(f"📊 Training curves saved to: training_curves_{experiment_name}.png")
    
    if use_wandb:
        wandb.log({"final_training_curves": wandb.Image(f'training_curves_{experiment_name}.png')})
        wandb.finish()
        print("✅ WandB logging completed")
    
    return controller, episode_rewards, collective_rewards, experiment_name


def test_trained_agents(model_path_prefix, env_name, n_episodes=5, visualize=True):
    """Test trained agents and optionally create visualization"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment
    env = gym.make(env_name)
    n_agents = env.n_agents
    
    # Create controller and load models
    controller = MultiAgentPPOController(env_name, n_agents, device, lr=3e-4, n_parallel_envs=1)
    controller.load_models(model_path_prefix)
    
    # Set agents to evaluation mode
    for agent in controller.agents:
        agent.eval()
    
    print(f"🧪 Testing trained agents in {env_name}")
    print(f"🤖 Number of agents: {n_agents}")
    
    test_results = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        total_rewards = [0] * n_agents
        steps = 0
        trajectory = []
        
        while True:
            # Store trajectory data for visualization
            if visualize:
                trajectory.append({
                    'obs': obs,
                    'full_image': env.render('rgb_array'),
                    'step': steps
                })
            
            # Get actions
            actions, _, _, _, _ = controller.get_actions(obs)
            
            # Execute actions
            obs, rewards, done, info = env.step(actions)
            
            for i in range(n_agents):
                total_rewards[i] += rewards[i]
            
            steps += 1
            
            if done or steps > 500:
                break
        
        collective_reward = sum(total_rewards)
        test_results.append({
            'episode': episode,
            'collective_reward': collective_reward,
            'individual_rewards': total_rewards,
            'steps': steps,
            'trajectory': trajectory if visualize else None
        })
        
        print(f"Test Episode {episode + 1}: "
              f"Collective = {collective_reward:.2f}, "
              f"Individual = {total_rewards}, "
              f"Steps = {steps}")
    
    env.close()
    
    avg_collective = np.mean([r['collective_reward'] for r in test_results])
    avg_steps = np.mean([r['steps'] for r in test_results])
    
    print(f"\n📊 Test Results Summary:")
    print(f"Average collective reward: {avg_collective:.2f}")
    print(f"Average episode length: {avg_steps:.1f}")
    
    return test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent PPO for MultiGrid - Final Deliverable")
    parser.add_argument("--env", type=str, default="MultiGrid-Cluttered-Fixed-15x15", 
                        help="Environment name")
    parser.add_argument("--episodes", type=int, default=1000, 
                        help="Number of training episodes (use 100000 for full training)")
    parser.add_argument("--parallel-envs", type=int, default=4, 
                        help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=128, 
                        help="Steps per rollout")
    parser.add_argument("--no-wandb", action="store_true", 
                        help="Disable WandB logging")
    parser.add_argument("--test", type=str, default=None, 
                        help="Test mode: provide model path prefix")
    parser.add_argument("--project", type=str, default="multigrid-multiagent-ppo", 
                        help="WandB project name")
    parser.add_argument("--name", type=str, default=None, 
                        help="Experiment name")
    
    args = parser.parse_args()
    
    if args.test:
        # Test mode
        print("🧪 Testing trained agents...")
        test_results = test_trained_agents(args.test, args.env)
    else:
        # Training mode
        print("🚀 Starting Multi-Agent PPO Training...")
        print("="*70)
        
        controller, episode_rewards, collective_rewards, experiment_name = train_multiagent_ppo(
            env_name=args.env,
            n_episodes=args.episodes,
            n_steps=args.steps,
            n_parallel_envs=args.parallel_envs,
            use_wandb=not args.no_wandb,
            project_name=args.project,
            experiment_name=args.name
        )
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Models saved with prefix: models/{experiment_name}")
        print(f"📊 Results saved to: final_results_{experiment_name}.json")
        print(f"📈 Training curves: training_curves_{experiment_name}.png")
        
        # Test the final model
        print("\n🧪 Testing final model...")
        test_results = test_trained_agents(f"models/{experiment_name}_final", args.env, n_episodes=3)
        
        print("\n🎉 Multi-Agent PPO implementation completed!")
        print("📝 Ready for submission with:")
        print("   ✅ WandB learning curves")
        print("   ✅ Model checkpoints") 
        print("   ✅ Test results")
        print("   ✅ Training visualizations")