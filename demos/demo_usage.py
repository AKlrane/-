"""
Example usage of the IndustryEnv environment.
Demonstrates how to use the multi-action space.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from config import Config
from env import IndustryEnv

def main():
    # Create environment with config
    config = Config()
    env = IndustryEnv(config.environment)
    
    # Reset with some initial firms
    obs, info = env.reset(seed=42, options={"initial_firms": 5})
    print("Initial observation:")
    print(f"  Number of firms: {obs['num_firms']}")
    print(f"  Total capital: ${obs['total_capital'][0]:,.2f}")
    print(f"  Sector distribution: {obs['sector_counts']}")
    print(f"  Average revenue: ${obs['avg_revenue'][0]:,.2f}\n")
    
    # Example 1: Use action space sampling
    print("Action 1: Random action (invest or create)")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Reward: {reward:.2f}")
    print(f"  Total profit: ${info['total_profit']:,.2f}")
    print(f"  Number of firms: {info['num_firms']}\n")
    
    # Example 2: Another random action
    print("Action 2: Another random action")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Reward: {reward:.2f}")
    print(f"  Total profit: ${info['total_profit']:,.2f}")
    print(f"  Number of firms: {info['num_firms']}")
    print(f"  Sector distribution: {obs['sector_counts']}\n")
    
    # Example 3: Random actions for a few steps
    print("Running 5 random actions:")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: reward={reward:.2f}, firms={info['num_firms']}")
    
    print(f"\nFinal state:")
    print(f"  Total capital: ${obs['total_capital'][0]:,.2f}")
    print(f"  Average revenue: ${obs['avg_revenue'][0]:,.2f}")
    print(f"  Sector distribution: {obs['sector_counts']}")
    
    # Example 4: Demonstrate location features
    print(f"\n--- Location Information ---")
    print(f"Companies and their locations:")
    for i, company in enumerate(env.companies[:5]):  # Show first 5 companies
        print(f"  Company {i}: Sector {company.sector_id}, Location ({company.x:.2f}, {company.y:.2f}), Capital ${company.capital:,.2f}")
    
    # Calculate distance between first two companies if they exist
    if len(env.companies) >= 2:
        distance = env.companies[0].distance_to(env.companies[1])
        print(f"\nDistance between Company 0 and Company 1: {distance:.2f} units")

if __name__ == "__main__":
    main()
