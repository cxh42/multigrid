"""
Fixed Multi-Agent PPO Trajectory Video Generator

Generate behavior trajectory videos for fixed agents, displaying:
- Environment state and agent movement
- Real-time reward changes  
- Behavior statistics
- Agent activity analysis
"""

import gym
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from collections import deque

# 导入修复版控制器
from v2_fixed_improved_multiagent_ppo import FixedMultiAgentPPOController, BehaviorMonitor
from envs import gym_multigrid
from envs.gym_multigrid import multigrid_envs


def create_fixed_agent_video(model_path, env_name, max_steps=150, output_dir="model2/videos"):
    """
    为修复版智能体创建轨迹视频
    
    Args:
        model_path: 模型路径前缀 (在model2文件夹中)
        env_name: 环境名称
        max_steps: 最大步数
        output_dir: 输出目录
        
    Returns:
        trajectory_data: 轨迹数据字典
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建环境
    env = gym.make(env_name)
    n_agents = env.n_agents
    
    print(f"🎬 Creating fixed agent trajectory video...")
    print(f"🤖 Number of agents: {n_agents}")
    print(f"📁 Model path: {model_path}")
    print(f"🎯 Environment: {env_name}")
    
    # 创建修复版控制器并加载模型
    controller = FixedMultiAgentPPOController(env_name, n_agents, device, n_parallel_envs=1)
    controller.load_models(model_path)
    
    # 设置为评估模式
    for agent in controller.agents:
        agent.eval()
    
    # 创建行为监控器
    behavior_monitor = BehaviorMonitor(n_agents)
    
    # 记录轨迹数据
    obs = env.reset()
    behavior_monitor.reset()
    
    total_rewards = [0] * n_agents
    all_images = []
    all_actions = []
    step_rewards = []
    behavior_history = []
    position_history = []
    entropy_history = []
    
    print("🎥 Starting trajectory recording...")
    
    for step in range(max_steps):
        # Render and store environment image
        img = env.render('rgb_array')
        all_images.append(img)
        
        # Get agent actions and additional info
        actions, log_probs, values, entropies, _ = controller.get_actions(obs)
        all_actions.append(actions)
        
        # Record entropy values (reflecting exploration level)
        avg_entropy = np.mean(entropies)
        entropy_history.append(avg_entropy)
        
        # Record positions
        current_positions = [tuple(pos) for pos in env.agent_pos]
        position_history.append(current_positions)
        
        # Update behavior monitoring
        behavior_monitor.update(actions, env.agent_pos)
        
        # Get current behavior metrics
        current_behavior = behavior_monitor.get_activity_metrics()
        behavior_history.append(current_behavior)
        
        # Execute actions
        obs, rewards, done, info = env.step(actions)
        
        # Track rewards
        for i in range(n_agents):
            total_rewards[i] += rewards[i]
        step_rewards.append(rewards)
        
        if done:
            print(f"🏁 Episode completed at step {step+1}")
            break
    
    actual_steps = len(all_images)
    print(f"✅ Recorded {actual_steps} frames")
    print(f"🏆 Final rewards: {total_rewards}")
    
    # Calculate final statistics
    final_behavior = behavior_monitor.get_activity_metrics()
    avg_move_ratio = final_behavior.get('avg_move_ratio', 0)
    avg_exploration = final_behavior.get('avg_exploration', 0)
    
    print(f"📊 Behavior statistics:")
    print(f"   Move ratio: {avg_move_ratio:.2f}")
    print(f"   Exploration range: {avg_exploration:.1f}")
    print(f"   Average entropy: {np.mean(entropy_history):.3f}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualization for each frame
    print("🖼️ Generating video frames...")
    
    for i in range(actual_steps):
        create_enhanced_frame(
            i, all_images[i], all_actions, step_rewards, total_rewards,
            behavior_history, entropy_history, position_history,
            n_agents, output_dir, env_name
        )
        
        if i % 20 == 0:
            print(f"   Generated frame {i}/{actual_steps}")
    
    env.close()
    
    # Generate FFmpeg command with 10fps
    video_name = f"trajectory_{env_name}_{os.path.basename(model_path)}"
    ffmpeg_cmd = f"ffmpeg -r 10 -i {output_dir}/frame_%03d.png -c:v libx264 -pix_fmt yuv420p {output_dir}/{video_name}.mp4"
    
    print(f"✅ All frames saved to: {output_dir}/")
    print(f"🎬 Video generation command:")
    print(f"   {ffmpeg_cmd}")
    
    # Try to automatically generate video
    try:
        import subprocess
        result = subprocess.run(ffmpeg_cmd.split(), capture_output=True, text=True)
        if result.returncode == 0:
            print(f"🎥 Video automatically generated: {output_dir}/{video_name}.mp4")
        else:
            print(f"⚠️ Auto video generation failed, please run the command manually")
    except Exception as e:
        print(f"⚠️ Please run FFmpeg command manually to generate video")
    
    # 返回轨迹数据
    trajectory_data = {
        'images': all_images,
        'actions': all_actions,
        'rewards': step_rewards,
        'total_rewards': total_rewards,
        'behavior_metrics': final_behavior,
        'entropy_history': entropy_history,
        'position_history': position_history,
        'steps': actual_steps
    }
    
    return trajectory_data


def create_enhanced_frame(frame_idx, env_image, all_actions, step_rewards, total_rewards,
                         behavior_history, entropy_history, position_history,
                         n_agents, output_dir, env_name):
    """Create enhanced video frame with comprehensive information"""
    
    plt.figure(figsize=(16, 12))
    
    # 1. Main environment image (occupies most of top-left space)
    plt.subplot(3, 4, (1, 6))
    plt.imshow(env_image)
    plt.title(f'Step {frame_idx} - {env_name}', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 2. Cumulative reward curves
    plt.subplot(3, 4, 3)
    if frame_idx > 0:
        for agent_id in range(n_agents):
            agent_rewards = [step_rewards[j][agent_id] for j in range(frame_idx)]
            cumulative = np.cumsum(agent_rewards)
            plt.plot(cumulative, label=f'Agent {agent_id}', linewidth=2)
    plt.title('Cumulative Rewards', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    
    # 3. Entropy changes (reflecting exploration level)
    plt.subplot(3, 4, 4)
    if len(entropy_history) > 0:
        plt.plot(entropy_history[:frame_idx+1], 'g-', linewidth=2)
        plt.axhline(y=np.mean(entropy_history[:frame_idx+1]), color='r', linestyle='--', alpha=0.7)
    plt.title('Exploration Level (Entropy)', fontweight='bold')
    plt.ylabel('Entropy')
    plt.xlabel('Steps')
    plt.grid(True, alpha=0.3)
    
    # 4. Current step information
    plt.subplot(3, 4, 7)
    plt.axis('off')
    
    # Action name mapping
    action_names = ['Left', 'Right', 'Forward', 'Pickup', 'Drop', 'Toggle', 'Wait']
    
    info_text = f"Step: {frame_idx}\n\n"
    if frame_idx < len(all_actions):
        for i in range(n_agents):
            action_idx = all_actions[frame_idx][i]
            action_name = action_names[action_idx] if action_idx < len(action_names) else f"Action{action_idx}"
            info_text += f"Agent{i}: {action_name}\n"
    
    info_text += f"\nCurrent Rewards:\n"
    if frame_idx < len(step_rewards):
        for i in range(n_agents):
            info_text += f"Agent{i}: {step_rewards[frame_idx][i]:.2f}\n"
    
    info_text += f"\nCumulative Total:\n"
    for i in range(n_agents):
        info_text += f"Agent{i}: {total_rewards[i]:.2f}\n"
    
    plt.text(0.05, 0.95, info_text, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # 5. Movement frequency statistics
    plt.subplot(3, 4, 8)
    if frame_idx < len(behavior_history) and behavior_history[frame_idx]:
        behavior = behavior_history[frame_idx]
        move_ratios = []
        agent_labels = []
        for i in range(n_agents):
            key = f'agent_{i}_move_ratio'
            if key in behavior:
                move_ratios.append(behavior[key])
                agent_labels.append(f'A{i}')
        
        if move_ratios:
            colors = ['lightgreen' if r > 0.6 else 'orange' if r > 0.3 else 'lightcoral' for r in move_ratios]
            plt.bar(agent_labels, move_ratios, color=colors, alpha=0.8)
            plt.title('Movement Frequency', fontweight='bold')
            plt.ylabel('Ratio')
            plt.ylim(0, 1)
            plt.axhline(y=0.6, color='g', linestyle='--', alpha=0.7, label='Target')
            plt.legend()
    
    # 6. Exploration heatmap (recently visited positions)
    plt.subplot(3, 4, 9)
    if frame_idx > 0 and position_history:
        # Create 15x15 visit heatmap
        heatmap = np.zeros((15, 15))
        
        # Count position visits in recent 30 steps
        start_idx = max(0, frame_idx - 30)
        for step_idx in range(start_idx, min(frame_idx + 1, len(position_history))):
            for pos in position_history[step_idx]:
                if 0 <= pos[0] < 15 and 0 <= pos[1] < 15:
                    heatmap[pos[1], pos[0]] += 1
        
        if np.max(heatmap) > 0:
            plt.imshow(heatmap, cmap='Reds', alpha=0.8, origin='upper')
            plt.title('Exploration Heatmap (Last 30 steps)', fontweight='bold')
            plt.colorbar(label='Visit Count')
        else:
            plt.text(0.5, 0.5, 'No Data', ha='center', va='center')
            plt.title('Exploration Heatmap', fontweight='bold')
    
    # 7. Real-time behavior metrics
    plt.subplot(3, 4, 10)
    plt.axis('off')
    
    behavior_text = "Real-time Behavior Analysis:\n\n"
    
    if frame_idx < len(behavior_history) and behavior_history[frame_idx]:
        behavior = behavior_history[frame_idx]
        
        avg_move = behavior.get('avg_move_ratio', 0)
        avg_explore = behavior.get('avg_exploration', 0)
        
        # Behavior assessment
        if avg_move > 0.7:
            move_status = "✅ Very Active"
        elif avg_move > 0.5:
            move_status = "⚡ Moderately Active"
        else:
            move_status = "⚠️ Low Activity"
        
        if avg_explore > 20:
            explore_status = "🗺️ Extensive Exploration"
        elif avg_explore > 10:
            explore_status = "🔍 Moderate Exploration"
        else:
            explore_status = "📍 Limited Exploration"
        
        behavior_text += f"Movement Activity: {avg_move:.2f}\n{move_status}\n\n"
        behavior_text += f"Exploration Range: {avg_explore:.1f}\n{explore_status}\n\n"
        
        if len(entropy_history) > frame_idx:
            current_entropy = entropy_history[frame_idx]
            if current_entropy > 1.5:
                entropy_status = "🎲 Highly Random"
            elif current_entropy > 1.0:
                entropy_status = "⚖️ Balanced Exploration"
            else:
                entropy_status = "🎯 Goal-Directed"
            
            behavior_text += f"Exploration Level: {current_entropy:.3f}\n{entropy_status}"
    else:
        behavior_text += "Computing..."
    
    plt.text(0.05, 0.95, behavior_text, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    # 8. Agent trajectory tracking
    plt.subplot(3, 4, 11)
    if position_history and frame_idx > 0:
        # Draw agent trajectories
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        for agent_id in range(n_agents):
            # Extract trajectory for this agent
            agent_positions = []
            for step_pos in position_history[:frame_idx+1]:
                if agent_id < len(step_pos):
                    agent_positions.append(step_pos[agent_id])
            
            if len(agent_positions) > 1:
                x_coords = [pos[0] for pos in agent_positions]
                y_coords = [pos[1] for pos in agent_positions]
                
                # Draw trajectory line
                plt.plot(x_coords, y_coords, color=colors[agent_id % len(colors)], 
                        alpha=0.6, linewidth=2, label=f'Agent {agent_id}')
                
                # Mark current position
                plt.scatter(x_coords[-1], y_coords[-1], color=colors[agent_id % len(colors)], 
                           s=100, marker='o', edgecolor='black', linewidth=2)
        
        plt.xlim(0, 14)
        plt.ylim(0, 14)
        plt.gca().invert_yaxis()  # Flip y-axis to match grid coordinates
        plt.title('Agent Trajectories', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 9. Overall performance assessment
    plt.subplot(3, 4, 12)
    plt.axis('off')
    
    # Calculate overall performance score
    total_collective_reward = sum(total_rewards)
    avg_entropy = np.mean(entropy_history[:frame_idx+1]) if entropy_history else 0
    
    if frame_idx < len(behavior_history) and behavior_history[frame_idx]:
        current_behavior = behavior_history[frame_idx]
        move_score = current_behavior.get('avg_move_ratio', 0) * 100
        explore_score = min(current_behavior.get('avg_exploration', 0) * 2, 100)
    else:
        move_score = 0
        explore_score = 0
    
    entropy_score = min(avg_entropy * 50, 100)  # Map entropy to 0-100
    
    # Overall score
    overall_score = (total_collective_reward + move_score + explore_score + entropy_score) / 4
    
    if overall_score > 80:
        performance_level = "🌟 Excellent"
        performance_color = "lightgreen"
    elif overall_score > 60:
        performance_level = "👍 Good"
        performance_color = "lightblue"
    elif overall_score > 40:
        performance_level = "⚡ Average"
        performance_color = "lightyellow"
    else:
        performance_level = "⚠️ Needs Improvement"
        performance_color = "lightcoral"
    
    summary_text = f"Performance Assessment:\n\n"
    summary_text += f"Collective Reward: {total_collective_reward:.1f}\n"
    summary_text += f"Movement Activity: {move_score:.0f}/100\n"
    summary_text += f"Exploration Breadth: {explore_score:.0f}/100\n"
    summary_text += f"Exploration Intensity: {entropy_score:.0f}/100\n\n"
    summary_text += f"Overall Score: {overall_score:.0f}/100\n"
    summary_text += f"Level: {performance_level}"
    
    plt.text(0.05, 0.95, summary_text, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor=performance_color, alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/frame_{frame_idx:03d}.png', dpi=120, bbox_inches='tight')
    plt.close()


def batch_create_videos(model_dir="model2", env_name="MultiGrid-Cluttered-Fixed-15x15", 
                       pattern="*_final", max_steps=150):
    """Batch generate videos for models in model2 folder"""
    
    import glob
    
    print(f"🎬 Batch video generation...")
    print(f"📁 Search directory: {model_dir}")
    print(f"🔍 Search pattern: {pattern}_agent_*.pth")
    
    # Find all matching models
    model_files = glob.glob(f"{model_dir}/{pattern}_agent_0.pth")
    model_prefixes = [f.replace("_agent_0.pth", "") for f in model_files]
    
    if not model_prefixes:
        print(f"❌ No matching model files found")
        return
    
    print(f"✅ Found {len(model_prefixes)} models:")
    for prefix in model_prefixes:
        print(f"   - {os.path.basename(prefix)}")
    
    # Generate video for each model
    results = []
    for i, model_prefix in enumerate(model_prefixes):
        print(f"\n🎥 Processing model {i+1}/{len(model_prefixes)}: {os.path.basename(model_prefix)}")
        
        try:
            trajectory_data = create_fixed_agent_video(
                model_prefix, env_name, max_steps, 
                output_dir=f"{model_dir}/videos/{os.path.basename(model_prefix)}"
            )
            results.append({
                'model': model_prefix,
                'success': True,
                'trajectory_data': trajectory_data
            })
            print(f"✅ Successfully generated video")
            
        except Exception as e:
            print(f"❌ Video generation failed: {e}")
            results.append({
                'model': model_prefix,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n📊 Batch generation completed:")
    print(f"   Successful: {successful}/{len(results)}")
    print(f"   Failed: {len(results) - successful}/{len(results)}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed Multi-Agent PPO Trajectory Video Generator")
    parser.add_argument("--model", type=str, 
                        help="Model path prefix (e.g., model2/experiment_name_final)")
    parser.add_argument("--env", type=str, default="MultiGrid-Cluttered-Fixed-15x15",
                        help="Environment name")
    parser.add_argument("--steps", type=int, default=150,
                        help="Maximum steps")
    parser.add_argument("--output", type=str, default="model2/videos",
                        help="Output directory")
    parser.add_argument("--batch", action="store_true",
                        help="Batch process all final models in model2 folder")
    parser.add_argument("--pattern", type=str, default="*_final",
                        help="Model file pattern for batch processing")
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch processing mode
        print("🎬 Batch video generation mode")
        results = batch_create_videos(
            model_dir="model2",
            env_name=args.env,
            pattern=args.pattern,
            max_steps=args.steps
        )
    else:
        # Single model processing
        if not args.model:
            print("❌ Please specify model path (--model) or use batch mode (--batch)")
            exit(1)
        
        print("🎬 Single model video generation mode")
        trajectory_data = create_fixed_agent_video(
            args.model, args.env, args.steps, args.output
        )
        
        print(f"\n🎉 Video generation completed!")
        print(f"📊 Trajectory statistics:")
        print(f"   Total steps: {trajectory_data['steps']}")
        print(f"   Final rewards: {trajectory_data['total_rewards']}")
        print(f"   Behavior metrics: {trajectory_data['behavior_metrics']}")