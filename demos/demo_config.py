"""
Example: Using config.json with the training infrastructure.
Demonstrates how to train an RL agent using hyperparameters from config.json.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from env import IndustryEnv
import numpy as np


def example_basic_usage():
    """Basic example: Create environment from config."""
    print("="*70)
    print("EXAMPLE 1: BASIC ENVIRONMENT CREATION FROM CONFIG")
    print("="*70)
    
    # Load configuration
    config = load_config("config.json")
    
    # Create environment using config object
    env = IndustryEnv(config.environment)
    
    # Reset with initial firms from config
    obs, info = env.reset(options={"initial_firms": config.environment.initial_firms})
    
    print(f"\n✅ Environment created successfully!")
    print(f"   Size: {config.environment.size}")
    print(f"   Max companies: {config.environment.max_company}")
    print(f"   Logistic cost rate: {config.environment.logistic_cost_rate}")
    print(f"   Initial firms: {obs['num_firms']}")
    print(f"   Total capital: ${obs['total_capital'][0]:,.2f}")
    
    return env, config


def example_run_episode():
    """Run a single episode using config parameters."""
    print("\n" + "="*70)
    print("EXAMPLE 2: RUNNING EPISODE WITH CONFIG REWARDS")
    print("="*70)
    
    config = load_config("config.json")
    env = IndustryEnv(config.environment)
    
    obs, _ = env.reset(options={"initial_firms": config.environment.initial_firms})
    
    # Run 10 steps with random actions
    total_reward = 0
    for step in range(10):
        # Use action space sampling
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step == 0:
            print(f"\n   Step 1:")
            print(f"      Actions taken: {info.get('num_actions', 0)}")
            print(f"      Valid actions: {info.get('num_valid_actions', 0)}")
            print(f"      Reward: {reward:.2f}")
            print(f"      Firms: {info['num_firms']}")
            print(f"      Profit: ${info['total_profit']:,.2f}")
            print(f"      Logistic cost: ${info['total_logistic_cost']:,.2f}")
    
    print(f"\n   After 10 steps:")
    print(f"      Total reward: {total_reward:.2f}")
    print(f"      Final firms: {obs['num_firms']}")
    print(f"      Final capital: ${obs['total_capital'][0]:,.2f}")


def example_ablation_study():
    """Example: Running ablation study (disable logistic costs)."""
    print("\n" + "="*70)
    print("EXAMPLE 3: ABLATION STUDY - COMPARE WITH/WITHOUT LOGISTIC COSTS")
    print("="*70)
    
    config = load_config("config.json")
    
    # Scenario 1: With logistic costs
    print("\n   🚚 Scenario 1: Logistic costs ENABLED")
    env1 = IndustryEnv(config.environment)
    obs1, _ = env1.reset(options={"initial_firms": 10})
    
    # Run simulation
    for _ in range(5):
        action = env1.action_space.sample()
        obs1, reward1, _, _, info1 = env1.step(action)
    
    print(f"      Total logistic cost: ${info1['total_logistic_cost']:,.2f}")
    print(f"      Total profit: ${info1['total_profit']:,.2f}")
    
    # Scenario 2: Without logistic costs (set rate to 0)
    print("\n   🚫 Scenario 2: Logistic costs DISABLED")
    config2 = load_config("config.json")
    config2.environment.logistic_cost_rate = 0.0
    env2 = IndustryEnv(config2.environment)
    obs2, _ = env2.reset(seed=42, options={"initial_firms": 10})
    
    # Run same simulation
    for _ in range(5):
        action = env2.action_space.sample()
        obs2, reward2, _, _, info2 = env2.step(action)
    
    print(f"      Total logistic cost: ${info2['total_logistic_cost']:,.2f}")
    print(f"      Total profit: ${info2['total_profit']:,.2f}")
    
    print(f"\n   📊 Impact of logistic costs:")
    print(f"      Profit difference: ${info1['total_profit'] - info2['total_profit']:,.2f}")


def example_hyperparameter_sweep():
    """Example: Test different logistic cost rates from config."""
    print("\n" + "="*70)
    print("EXAMPLE 4: HYPERPARAMETER SWEEP - LOGISTIC COST RATES")
    print("="*70)
    
    config = load_config("config.json")
    
    # Test different rates
    rates = [10.0, 100.0, 500.0, 1000.0]
    
    print(f"\n   Testing {len(rates)} different logistic cost rates:")
    print(f"   Base rate from config: {config.environment.logistic_cost_rate}\n")
    
    print(f"   {'Rate':>8} | {'Total Cost':>12} | {'Avg Cost/Firm':>15} | {'Total Profit':>12}")
    print("   " + "-"*60)
    
    for rate in rates:
        test_config = load_config("config.json")
        test_config.environment.logistic_cost_rate = rate
        env = IndustryEnv(test_config.environment)
        obs, _ = env.reset(options={"initial_firms": 10})
        
        # Run a few steps
        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, _, _, info = env.step(action)
        
        avg_cost = info['total_logistic_cost'] / info['num_firms']
        print(f"   {rate:>8.1f} | ${info['total_logistic_cost']:>11.2f} | ${avg_cost:>14.2f} | ${info['total_profit']:>11.2f}")
    
    print("\n   💡 Lower rates = lower costs = higher profits")


def example_save_custom_config():
    """Example: Modify and save custom configuration."""
    print("\n" + "="*70)
    print("EXAMPLE 5: CREATING CUSTOM CONFIGURATIONS")
    print("="*70)
    
    config = load_config("config.json")
    
    # Scenario 1: High-cost configuration
    print("\n   Creating 'config_high_cost.json'...")
    config.environment.logistic_cost_rate = 1000.0
    config.training.total_timesteps = 200000
    config.training.learning_rate = 0.0001
    config.to_json("config_high_cost.json")
    print("   ✅ Saved with logistic_cost_rate=1000.0")
    
    # Scenario 2: Fast training configuration
    print("\n   Creating 'config_fast_training.json'...")
    config = load_config("config.json")  # Reload defaults
    config.training.total_timesteps = 50000
    config.training.num_envs = 8
    config.environment.initial_firms = 5
    config.to_json("config_fast_training.json")
    print("   ✅ Saved with 50k timesteps, 8 envs, 5 initial firms")
    
    # Scenario 3: Research/ablation configuration
    print("\n   Creating 'demo_config_high_cost.json'...")
    config = load_config("config/config.json")
    config.environment.logistic_cost_rate = 1000.0
    config.to_json("config/demo_config_high_cost.json")
    print("   ✅ Saved with logistic_cost_rate=1000.0")

    print("\n   Creating 'demo_config_fast_training.json'...")
    config = load_config("config/config.json")
    config.training.timesteps = 50000
    config.training.num_envs = 8
    config.environment.initial_firms = 5
    config.to_json("config/demo_config_fast_training.json")
    print("   ✅ Saved with 50k timesteps, 8 envs, 5 initial firms")

    # Scenario 3: Research/ablation configuration
    print("\n   Creating 'demo_config_ablation.json'...")
    config = load_config("config/config.json")
    config.environment.disable_logistic_costs = True
    config.environment.logistic_cost_rate = 0.0
    config.to_json("config/demo_config_ablation.json")
    print("   ✅ Saved with logistic costs disabled")

    print("\n   \U0001F4C1 Custom configurations created in config/ folder!")
    print("   Use them with: python main.py --config config/demo_config_high_cost.json")
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
✅ Configuration system ready!

Key features:
1. Centralized hyperparameter management in config.json
2. Easy loading with load_config()
3. Organized into logical sections (environment, training, etc.)
4. Support for ablation studies
5. Save custom configurations for different experiments

Usage in your code:
    from config import load_config
    
    config = load_config("config/config.json")
    env = IndustryEnv(config.environment)

Experiment with different hyperparameters by editing config.json!
""")
