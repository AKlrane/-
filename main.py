"""
Main training script for Industry Simulation RL agents.
Supports Stable-Baselines3 framework and custom training loops.
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from typing import Optional

from env import IndustryEnv
from utils import create_dashboard
from config import load_config, Config
import matplotlib.pyplot as plt


def create_env(config: Config, seed: Optional[int] = None):
    """Create a single environment instance."""
    env = IndustryEnv(config.environment)
    if seed is not None:
        env.reset(seed=seed, options={"initial_firms": config.environment.initial_firms})
    return env


def train_stable_baselines3(config: Config):
    """Train using Stable-Baselines3."""
    try:
        from stable_baselines3 import PPO, A2C, SAC
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("ERROR: Stable-Baselines3 not installed!")
        print("Install with: pip install stable-baselines3")
        return
    
    # Create vectorized environment
    def make_env(i):
        def _init():
            env = create_env(config, seed=config.training.seed + i)
            env = Monitor(env, filename=os.path.join(config.training.log_dir, f"env_{i}"))
            return env
        return _init
    
    vec_env = DummyVecEnv([make_env(i) for i in range(config.training.num_envs)])
    
    # Create evaluation environment
    eval_env = Monitor(create_env(config, seed=config.training.seed + 1000))
    
    # Select algorithm
    algorithm_map = {
        "ppo": PPO,
        "a2c": A2C,
    }
    
    if config.training.algorithm not in algorithm_map:
        print(f"Unknown algorithm: {config.training.algorithm}. Using PPO.")
        config.training.algorithm = "ppo"
    
    AlgorithmClass = algorithm_map[config.training.algorithm]
    
    # Configure large MLP network architecture (~20-30 MB model)
    import torch.nn as nn
    
    policy_kwargs = dict(
        net_arch=dict(
            pi=[1024, 1024, 512, 512, 256],  # Actor: 5 layers
            vf=[1024, 1024, 512, 512, 256]   # Critic: 5 layers
        ),
        activation_fn=nn.ReLU,
    )
    
    print(f"🧠 Network: [1024×5] → ~20-30 MB | Envs: {config.training.num_envs} | LR: {config.training.learning_rate}")
    
    # Determine device (GPU if available)
    import torch
    device = config.training.device
    
    # Handle "auto" device selection
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("💻 Using CPU (no GPU detected)")
    else:
        # Manual device specification
        if "cuda" in device and torch.cuda.is_available():
            gpu_id = 0 if device == "cuda" else int(device.split(":")[1])
            print(f"🎮 Using GPU: {torch.cuda.get_device_name(gpu_id)} ({device})")
        elif device == "cpu":
            print("💻 Using CPU (manual)")
        else:
            print(f"⚠️  Device '{device}' not available, falling back to CPU")
            device = "cpu"
    
    # Create model
    model = AlgorithmClass(
        "MultiInputPolicy",
        vec_env,
        learning_rate=config.training.learning_rate,
        n_steps=config.training.n_steps,
        batch_size=config.training.batch_size,
        gamma=config.training.gamma,
        verbose=1,
        tensorboard_log=config.training.log_dir,
        seed=config.training.seed,
        policy_kwargs=policy_kwargs,
        device=device
    )
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config.training.save_freq // config.training.num_envs,
        save_path=config.training.checkpoint_dir,
        name_prefix=f"{config.training.algorithm}_model"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(config.training.checkpoint_dir, "best_model"),
        log_path=config.training.log_dir,
        eval_freq=config.training.eval_freq,
        deterministic=True,
        render=False
    )
    
    # Training
    print("\n⏳ Starting training...\n")
    
    start_time = time.time()
    
    try:
        # Try to use progress bar with tqdm
        try:
            model.learn(
                total_timesteps=config.training.total_timesteps,
                callback=[checkpoint_callback, eval_callback],
                progress_bar=True
            )
        except ImportError:
            # If tqdm not available, train without progress bar
            print("⚠️  tqdm not available, training without progress bar...")
            model.learn(
                total_timesteps=config.training.total_timesteps,
                callback=[checkpoint_callback, eval_callback],
                progress_bar=False
            )
    except KeyboardInterrupt:
        print("\n\n⛔ Training interrupted by user.")
    
    training_time = time.time() - start_time
    
    # Save final model
    final_path = os.path.join(config.training.checkpoint_dir, f"{config.training.algorithm}_final")
    model.save(final_path)
    
    print("\n" + "="*70)
    print(f"✅ Training completed in {training_time/60:.1f} minutes ({training_time/3600:.2f} hours)")
    print(f"💾 Model saved: {final_path}")
    print("="*70)
    
    # Evaluate final model
    print("\nEvaluating final model...")
    evaluate_model_sb3(model, config, num_episodes=10)
    
    vec_env.close()
    eval_env.close()
    
    return model


def train_custom(config: Config):
    """Custom training loop (for educational purposes or custom algorithms)."""
    print("="*60)
    print("Custom Training Loop")
    print("="*60)
    
    env = create_env(config, seed=config.training.seed)
    
    print(f"\nEnvironment created")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Initialize simple random policy for demonstration
    print("\nRunning random policy for demonstration...")
    print("(Implement your custom algorithm here)")
    
    episode_rewards = []
    num_episodes = config.training.total_timesteps // 1000  # Approximate
    
    start_time = time.time()
    
    initial_firms = config.environment.initial_firms
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=config.training.seed + episode, options={"initial_firms": initial_firms})
        done = False
        episode_reward = 0
        step = 0
        
        while not done and step < 1000:
            # Random action (replace with your policy)
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step += 1
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{num_episodes} - Avg Reward: {avg_reward:.2f} - Steps: {step} - Firms: {info['num_firms']}")
        
        # Visualize periodically
        if (episode + 1) % (config.training.visualize_freq // 1000) == 0:
            fig = create_dashboard(env)
            if fig is not None:
                fig_path = os.path.join(config.training.log_dir, f"dashboard_episode_{episode+1}.png")
                fig.savefig(fig_path, dpi=config.environment.dpi, bbox_inches='tight')
                plt.close(fig)
                print(f"  Dashboard saved to {fig_path}")
    
    training_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Training time: {training_time/60:.2f} minutes")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Final episode firms: {info['num_firms']}")
    
    # Save training history
    history_path = os.path.join(config.training.log_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump({
            "episode_rewards": episode_rewards,
            "num_episodes": num_episodes,
            "training_time": training_time,
            "config": vars(config)
        }, f, indent=2)
    
    print(f"Training history saved to {history_path}")
    
    return episode_rewards


def evaluate_model_sb3(model, config: Config, num_episodes: int = 10):
    """Evaluate a trained Stable-Baselines3 model."""
    from stable_baselines3.common.monitor import Monitor
    
    eval_env = Monitor(create_env(config, seed=config.training.seed + 9999))
    initial_firms = config.environment.initial_firms
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = eval_env.reset(seed=config.training.seed + episode + 10000, options={"initial_firms": initial_firms})
        done = False
        episode_reward = 0
        step = 0
        
        while not done and step < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        
        print(f"  Episode {episode+1}: Reward={episode_reward:.2f}, Steps={step}, Firms={info['num_firms']}")
    
    print(f"\nEvaluation Results:")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    # Visualize final episode
    fig = create_dashboard(eval_env.unwrapped)
    if fig is not None:
        fig_path = os.path.join(config.training.log_dir, "evaluation_dashboard.png")
        fig.savefig(fig_path, dpi=config.environment.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  Evaluation dashboard saved to {fig_path}")
    
    eval_env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train RL agents for Industry Simulation")
    
    # Configuration file
    parser.add_argument("--config", type=str, default="config/config.json",
                       help="Path to configuration file (default: config/config.json)")
    
    # Framework and algorithm (can override config)
    parser.add_argument("--framework", type=str, default=None,
                       help="RL framework to use (overrides config)")
    parser.add_argument("--algorithm", type=str, default=None,
                       help="RL algorithm (overrides config)")
    
    # Training parameters (can override config)
    parser.add_argument("--timesteps", type=int, default=None,
                       help="Total training timesteps (overrides config)")
    parser.add_argument("--num-envs", type=int, default=None,
                       help="Number of parallel environments (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--gamma", type=float, default=None,
                       help="Discount factor (overrides config)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed (overrides config)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use: 'auto', 'cuda', 'cpu', 'cuda:0', etc. (overrides config)")
    
    # Environment parameters (can override config)
    parser.add_argument("--env-size", type=float, default=None,
                       help="Environment spatial size (overrides config)")
    parser.add_argument("--max-company", type=int, default=None,
                       help="Maximum number of companies (overrides config)")
    parser.add_argument("--initial-firms", type=int, default=None,
                       help="Initial number of firms (overrides config)")
    parser.add_argument("--logistic-cost-rate", type=float, default=None,
                       help="Logistic cost rate (overrides config)")
    parser.add_argument("--revenue-rate", type=float, default=None,
                       help="Revenue rate (overrides config)")
    parser.add_argument("--death-threshold", type=float, default=None,
                       help="Company death threshold (overrides config)")
    
    # Logging and checkpoints (can override config)
    parser.add_argument("--log-dir", type=str, default=None,
                       help="Directory for logs (overrides config)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                       help="Directory for checkpoints (overrides config)")
    parser.add_argument("--save-freq", type=int, default=None,
                       help="Save checkpoint every N steps (overrides config)")
    parser.add_argument("--eval-freq", type=int, default=None,
                       help="Evaluate every N steps (overrides config)")
    parser.add_argument("--visualize-freq", type=int, default=None,
                       help="Create visualization every N steps (overrides config)")
    
    # Modes
    parser.add_argument("--mode", type=str, default="train",
                       choices=["train", "eval", "demo"],
                       help="Mode: train, eval, or demo (default: train)")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to model for evaluation or demo")
    
    args = parser.parse_args()
    
    # Load configuration from file
    print(f"\nLoading configuration from: {args.config}")
    global_config = load_config(args.config)
    
    # Apply command-line overrides
    if args.framework is not None:
        global_config.training.framework = args.framework
    if args.algorithm is not None:
        global_config.training.algorithm = args.algorithm
    if args.timesteps is not None:
        global_config.training.total_timesteps = args.timesteps
    if args.num_envs is not None:
        global_config.training.num_envs = args.num_envs
    if args.lr is not None:
        global_config.training.learning_rate = args.lr
    if args.batch_size is not None:
        global_config.training.batch_size = args.batch_size
    if args.gamma is not None:
        global_config.training.gamma = args.gamma
    if args.seed is not None:
        global_config.training.seed = args.seed
    if args.device is not None:
        global_config.training.device = args.device

    # Environment overrides
    if args.env_size is not None:
        global_config.environment.size = args.env_size
    if args.max_company is not None:
        global_config.environment.max_company = args.max_company
    if args.initial_firms is not None:
        global_config.environment.initial_firms = args.initial_firms
    if args.logistic_cost_rate is not None:
        global_config.environment.logistic_cost_rate = args.logistic_cost_rate
    if args.revenue_rate is not None:
        global_config.environment.revenue_rate = args.revenue_rate
    if args.death_threshold is not None:
        global_config.environment.death_threshold = args.death_threshold
    
    # Checkpoint overrides (now in training config)
    if args.log_dir is not None:
        global_config.training.log_dir = args.log_dir
    if args.checkpoint_dir is not None:
        global_config.training.checkpoint_dir = args.checkpoint_dir
    if args.save_freq is not None:
        global_config.training.save_freq = args.save_freq
    if args.eval_freq is not None:
        global_config.training.eval_freq = args.eval_freq
    if args.visualize_freq is not None:
        global_config.training.visualize_freq = args.visualize_freq
    
    # Configuration loaded (simplified output)
    
    # Create timestamp-based directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    framework = global_config.training.framework
    algorithm = global_config.training.algorithm
    
    # Update log and checkpoint directories with timestamp
    base_log_dir = global_config.training.log_dir
    base_checkpoint_dir = global_config.training.checkpoint_dir
    global_config.training.log_dir = os.path.join(base_log_dir, f"{framework}_{algorithm}_{timestamp}")
    global_config.training.checkpoint_dir = os.path.join(base_checkpoint_dir, f"{framework}_{algorithm}_{timestamp}")
    
    # Create directories
    Path(global_config.training.log_dir).mkdir(parents=True, exist_ok=True)
    Path(global_config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Print simplified training info
    print("\n" + "="*70)
    print(f"🚀 Training: {global_config.training.algorithm.upper()} | {global_config.training.total_timesteps:,} steps | Seed {global_config.training.seed}")
    print(f"📁 Logs: {global_config.training.log_dir}")
    print("="*70 + "\n")
    
    # Save config
    config_path = os.path.join(global_config.training.log_dir, "config.json")
    global_config.to_json(config_path)
    
    # Execute based on mode
    if args.mode == "train":
        # Training
        if global_config.training.framework == "sb3":
            model = train_stable_baselines3(global_config)
        elif global_config.training.framework == "custom":
            model = train_custom(global_config)
        else:
            print(f"Unknown framework: {global_config.training.framework}. Only 'sb3' and 'custom' are supported.")
            return
    
    elif args.mode == "eval":
        # Evaluation
        if args.model_path is None:
            print("ERROR: --model-path required for evaluation mode")
            return
        
        if global_config.training.framework == "sb3":
            from stable_baselines3 import PPO, A2C
            algorithm_map = {"ppo": PPO, "a2c": A2C}
            AlgorithmClass = algorithm_map[global_config.training.algorithm]
            model = AlgorithmClass.load(args.model_path)
            evaluate_model_sb3(model, global_config, num_episodes=20)
        else:
            print(f"Evaluation not yet implemented for {global_config.training.framework}")
    
    elif args.mode == "demo":
        # Demo mode - visualize a trained agent
        print("Demo mode - running trained agent with visualization")
        if args.model_path is None:
            print("WARNING: No model path provided, using random policy")
        
        initial_firms = global_config.environment.initial_firms
        env = create_env(global_config, seed=global_config.training.seed)
        obs, info = env.reset(seed=global_config.training.seed, options={"initial_firms": initial_firms})
        
        for step in range(100):
            if args.model_path and global_config.training.framework == "sb3":
                from stable_baselines3 import PPO, A2C
                algorithm_map = {"ppo": PPO, "a2c": A2C}
                AlgorithmClass = algorithm_map.get(global_config.training.algorithm, PPO)
                model = AlgorithmClass.load(args.model_path)
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step % 10 == 0:
                print(f"Step {step}: Reward={reward:.2f}, Firms={info['num_firms']}, Action={info['action_type']}")
            
            if terminated or truncated:
                break
        
        # Show final visualization
        fig = create_dashboard(env)
        plt.show()
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
