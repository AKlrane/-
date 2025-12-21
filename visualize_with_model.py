"""
Visualize trained model performance by comparing environments with and without model intervention.
Creates side-by-side comparison images showing the difference between natural evolution and model-guided investment.
"""

import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import time

from env import IndustryEnv
from config import load_config
from main import MapObservationWrapper, RewardScaleWrapper
from utils.visualize import create_comparison_dashboard


def visualize_with_model(
    model_path: str,
    output_dir: str = "visualizations/model_comparison",
    steps: int = 64,
    save_every: int = 5,
    seed: int = None
):
    """
    Visualize model performance by comparing environments with and without model intervention.
    
    Args:
        model_path: Path to the trained model (e.g., "./checkpoints/cnn_ppo_xxx/cnn_ppo_final.zip")
        output_dir: Directory to save comparison images
        steps: Total number of steps to run (default: 64)
        save_every: Save image every N steps (default: 5)
        seed: Random seed for environment initialization (default: None for random)
    """
    # Load model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    print("Model loaded successfully!")
    
    # Load config
    config = load_config("config/config.json")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize random seed - use time-based seed if not provided for true randomness
    if seed is None:
        # Use current time in microseconds for better randomness
        seed = int(time.time() * 1000000) % (2**31)
        print(f"\nNo seed provided, using random seed based on current time")
    else:
        print(f"\nUsing provided seed for reproducible results")
    
    print(f"Seed: {seed}")
    print(f"Running {steps} steps, saving every {save_every} steps")
    print(f"Output directory: {output_dir}")
    print(f"Note: To reproduce this run, use seed={seed}\n")
    
    # Create two environments with same initial state
    # Environment 1: Random actions (uniform distribution between invest and create)
    env_no_action = IndustryEnv(config.environment)
    # Use config's fixed_investment_amount (80000) for random investment actions
    env_no_action.reset(seed=seed, options={"initial_firms": config.environment.initial_firms})
    
    # Environment 2: With model (needs wrapper for observation) - keep config's 80000
    env_with_model = IndustryEnv(config.environment)
    # fixed_investment_amount remains 80000.0 from config for model group
    env_with_model.reset(seed=seed, options={"initial_firms": config.environment.initial_firms})
    # 保存MapObservationWrapper的引用，用于获取observation
    env_with_model_obs_wrapper = MapObservationWrapper(env_with_model)
    env_with_model_wrapped = RewardScaleWrapper(env_with_model_obs_wrapper, scale=0.1)
    
    # 获取初始observation（通过MapObservationWrapper）
    obs_with_model = env_with_model_obs_wrapper.observation(env_with_model._get_observation())
    
    # Initialize random number generator for control group actions
    rng = np.random.RandomState(seed + 10000)  # Use different seed from environment
    
    # Save initial state
    print("Saving initial state (step 0)...")
    from utils.visualize import create_dashboard
    fig_no_action = create_dashboard(env_no_action, figsize=(18, 12))
    if fig_no_action is not None:
        fig_no_action.savefig(os.path.join(output_dir, f"no_action_step_{0:03d}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig_no_action)
        print(f"  Saved: no_action_step_{0:06d}.png")
    else:
        print("  Warning: Failed to create dashboard for no_action")
    
    fig_with_model = create_dashboard(env_with_model, figsize=(18, 12))
    if fig_with_model is not None:
        fig_with_model.savefig(os.path.join(output_dir, f"with_model_step_{0:06d}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig_with_model)
        print(f"  Saved: with_model_step_{0:06d}.png")
    else:
        print("  Warning: Failed to create dashboard for with_model")
    
    # Run simulation
    for step in range(1, steps + 1):
        # Environment 1: Random actions
        # Generate random action with logits for action type only (tier is determined by env rotation)
        # Action space: [logits_invest, logits_create, x, y]
        random_action = np.array([
            rng.normal(0.0, 1.0),  # logits_invest
            rng.normal(0.0, 1.0),  # logits_create
            rng.uniform(-1.0, 1.0),  # x coordinate (normalized to [-1, 1])
            rng.uniform(-1.0, 1.0)  # y coordinate (normalized to [-1, 1])
        ], dtype=np.float32)
        obs_no_action, reward_no_action, terminated_no, truncated_no, info_no = env_no_action.step(random_action)
        
        # Environment 2: Use model to predict action
        # 使用MapObservationWrapper获取observation
        obs_with_model = env_with_model_obs_wrapper.observation(env_with_model._get_observation())
        action_with_model, _ = model.predict(obs_with_model, deterministic=True)
        
        # Step环境（使用原始环境，因为action不需要wrapper处理）
        obs_with_model_raw, reward_with_model, terminated_with, truncated_with, info_with = env_with_model.step(action_with_model)
        
        # 如果环境被重置了，需要重新获取observation
        if terminated_with or truncated_with:
            env_with_model.reset(seed=seed, options={"initial_firms": config.environment.initial_firms})
            obs_with_model = env_with_model_obs_wrapper.observation(env_with_model._get_observation())
        
        # Save separate images every save_every steps
        if step % save_every == 0:
            print(f"Saving step {step}...")
            # 保存无干预环境的图片
            from utils.visualize import create_dashboard
            fig_no_action = create_dashboard(env_no_action, figsize=(18, 12))
            if fig_no_action is not None:
                fig_no_action.savefig(os.path.join(output_dir, f"no_action_step_{step:03d}.png"), dpi=150, bbox_inches='tight')
                plt.close(fig_no_action)
                print(f"  Saved: no_action_step_{step:06d}.png")
            else:
                print("  Warning: Failed to create dashboard for no_action")
            
            # 保存模型干预环境的图片
            fig_with_model = create_dashboard(env_with_model, figsize=(18, 12))
            if fig_with_model is not None:
                fig_with_model.savefig(os.path.join(output_dir, f"with_model_step_{step:03d}.png"), dpi=150, bbox_inches='tight')
                plt.close(fig_with_model)
                print(f"  Saved: with_model_step_{step:06d}.png")
            else:
                print("  Warning: Failed to create dashboard for with_model")
            
            # Print statistics
            total_capital_na = sum(c.capital for c in env_no_action.companies)
            total_capital_wm = sum(c.capital for c in env_with_model.companies)
            print(f"  No intervention - Companies: {len(env_no_action.companies)}, Capital: ${total_capital_na:,.0f}")
            print(f"  Model investment - Companies: {len(env_with_model.companies)}, Capital: ${total_capital_wm:,.0f}")
    
    print(f"\nVisualization complete! Images saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_with_model.py <model_path> [output_dir] [steps] [save_every] [seed]")
        print("\nExample:")
        print("  python visualize_with_model.py ./checkpoints/cnn_ppo_20251219_210300/cnn_ppo_final.zip")
        print("  python visualize_with_model.py ./checkpoints/cnn_ppo_20251219_210300/best_model/best_model.zip visualizations/model_comparison 64 5 42")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "visualizations/model_comparison"
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    save_every = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else None
    
    visualize_with_model(model_path, output_dir, steps, save_every, seed)
