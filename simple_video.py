"""
Simple test to create a basic trajectory video without using utils.py
"""

import gym
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from final_multiagent_ppo import MultiAgentPPOController
from envs import gym_multigrid
from envs.gym_multigrid import multigrid_envs

def create_simple_video(model_path, env_name, max_steps=100):
    """Create a simple trajectory visualization"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment
    env = gym.make(env_name)
    n_agents = env.n_agents
    
    print(f"🎬 Creating simple trajectory video...")
    print(f"🤖 Agents: {n_agents}")
    
    # Create controller and load models
    controller = MultiAgentPPOController(env_name, n_agents, device, lr=3e-4, n_parallel_envs=1)
    controller.load_models(model_path)
    
    # Set to evaluation mode
    for agent in controller.agents:
        agent.eval()
    
    # Record trajectory
    obs = env.reset()
    total_rewards = [0] * n_agents
    all_images = []
    all_actions = []
    step_rewards = []
    
    print("🎥 Recording...")
    
    for step in range(max_steps):
        # Render and store image
        img = env.render('rgb_array')
        all_images.append(img)
        
        # Get actions
        actions, _, _, _, _ = controller.get_actions(obs)
        all_actions.append(actions)
        
        # Execute
        obs, rewards, done, info = env.step(actions)
        
        # Track rewards
        for i in range(n_agents):
            total_rewards[i] += rewards[i]
        step_rewards.append(rewards)
        
        if done:
            break
    
    print(f"✅ Recorded {len(all_images)} frames")
    print(f"🏆 Final rewards: {total_rewards}")
    
    # Create simple visualization
    output_dir = "simple_videos"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual frames
    for i, img in enumerate(all_images):
        plt.figure(figsize=(10, 8))
        
        # Main environment image
        plt.subplot(2, 2, (1, 2))
        plt.imshow(img)
        plt.title(f'Step {i} - Environment')
        plt.axis('off')
        
        # Rewards plot
        plt.subplot(2, 2, 3)
        if i > 0:
            for agent_id in range(n_agents):
                agent_rewards = [step_rewards[j][agent_id] for j in range(i)]
                cumulative = np.cumsum(agent_rewards)
                plt.plot(cumulative, label=f'Agent {agent_id}')
        plt.title('Cumulative Rewards')
        plt.legend()
        plt.grid(True)
        
        # Current step info
        plt.subplot(2, 2, 4)
        plt.text(0.1, 0.8, f'Step: {i}', fontsize=14)
        if i < len(all_actions):
            plt.text(0.1, 0.6, f'Actions: {all_actions[i]}', fontsize=12)
        plt.text(0.1, 0.4, f'Rewards: {step_rewards[i] if i < len(step_rewards) else [0]*n_agents}', fontsize=12)
        plt.text(0.1, 0.2, f'Total: {total_rewards}', fontsize=12)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/frame_{i:03d}.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        if i % 20 == 0:
            print(f"Generated frame {i}")
    
    print(f"✅ Frames saved to {output_dir}/")
    print(f"🎬 To create video: ffmpeg -r 5 -i {output_dir}/frame_%03d.png -c:v libx264 trajectory.mp4")
    
    env.close()
    return total_rewards

if __name__ == "__main__":
    model_path = "models/MultiGrid-Cluttered-Fixed-15x15_3agents_1749607927_final"
    env_name = "MultiGrid-Cluttered-Fixed-15x15"
    
    rewards = create_simple_video(model_path, env_name)
    print(f"\n🎉 Complete! Final rewards: {rewards}")