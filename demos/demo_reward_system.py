#!/usr/bin/env python3
"""
Example demonstrating the new isolated reward calculation system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from env import IndustryEnv


def demo_reward_system():
    """Demonstrate how the reward system works."""
    
    print("="*70)
    print("REWARD SYSTEM DEMONSTRATION")
    print("="*70)
    
    # Create environment
    config = Config()
    config.environment.investment_multiplier = 0.01
    config.environment.creation_reward = 50.0
    config.environment.invalid_action_penalty = -20.0
    config.environment.invalid_firm_penalty = -10.0
    config.environment.profit_multiplier = 0.001
    
    env = IndustryEnv(config.environment)
    obs, _ = env.reset(seed=42, options={'initial_firms': 5})
    
    print(f"\nStarting with {obs['num_firms']} firms")
    print(f"\nReward parameters:")
    print(f"  - Investment multiplier: {config.environment.investment_multiplier}")
    print(f"  - Creation reward: {config.environment.creation_reward}")
    print(f"  - Invalid action penalty: {config.environment.invalid_action_penalty}")
    print(f"  - Invalid firm penalty: {config.environment.invalid_firm_penalty}")
    print(f"  - Profit multiplier: {config.environment.profit_multiplier}")
    
    # Test 1: Valid investment (single action in multi-action format)
    print("\n" + "="*70)
    print("TEST 1: Valid Investment (Single Action)")
    print("="*70)
    
    # Helper to create dummy actions
    dummy_action = {
        'op': 0,
        'invest': {'firm_id': 0, 'amount': [0.0]},
        'create': {'sector': 0, 'initial_capital': [10000.0], 'location': [50.0, 50.0]}
    }
    
    action = {
        'num_actions': 1,
        'actions': [
            {
                'op': 0,
                'invest': {'firm_id': 0, 'amount': [1000.0]},
                'create': {'sector': 0, 'initial_capital': [10000.0], 'location': [50.0, 50.0]}
            }
        ] + [dummy_action] * (env.max_actions_per_step - 1)
    }
    obs, reward, _, _, info = env.step(action)
    
    print(f"\nAction: Invest $1,000 in firm 0")
    print(f"Number of actions: {info['num_actions']}")
    print(f"Valid actions: {info['num_valid_actions']}")
    action_result = info['action_results'][0]
    print(f"Result: {action_result['result']}")
    print(f"Investment amount: ${action_result.get('investment_amount', 0):.2f}")
    print(f"Total profit: ${info['total_profit']:.2f}")
    print(f"\nReward breakdown:")
    investment_reward = action_result.get('investment_amount', 0) * config.environment.investment_multiplier
    profit_reward = info['total_profit'] * config.environment.profit_multiplier
    print(f"  Action reward:  ${investment_reward:>8.2f} (investment × multiplier)")
    print(f"  Profit reward:  ${profit_reward:>8.2f} (profit × multiplier)")
    print(f"  Total reward:   ${reward:>8.2f}")
    
    # Test 2: Valid creation
    print("\n" + "="*70)
    print("TEST 2: Valid Company Creation (Single Action)")
    print("="*70)
    action = {
        'num_actions': 1,
        'actions': [
            {
                'op': 1,
                'invest': {'firm_id': 0, 'amount': [1000.0]},
                'create': {'sector': 0, 'initial_capital': [50000.0], 'location': [50.0, 50.0]}
            }
        ] + [dummy_action] * (env.max_actions_per_step - 1)
    }
    obs, reward, _, _, info = env.step(action)
    
    print(f"\nAction: Create new company with $50,000 capital")
    action_result = info['action_results'][0]
    print(f"Result: {action_result['result']}")
    print(f"New firm count: {info['num_firms']}")
    print(f"Total profit: ${info['total_profit']:.2f}")
    print(f"\nReward breakdown:")
    creation_reward = config.environment.creation_reward
    profit_reward = info['total_profit'] * config.environment.profit_multiplier
    print(f"  Creation reward: ${creation_reward:>8.2f} (fixed creation bonus)")
    print(f"  Profit reward:   ${profit_reward:>8.2f} (profit × multiplier)")
    print(f"  Total reward:    ${reward:>8.2f}")
    
    # Test 3: Invalid firm ID
    print("\n" + "="*70)
    print("TEST 3: Invalid Firm ID (Single Action)")
    print("="*70)
    action = {
        'num_actions': 1,
        'actions': [
            {
                'op': 0,
                'invest': {'firm_id': 999, 'amount': [1000.0]},
                'create': {'sector': 0, 'initial_capital': [10000.0], 'location': [50.0, 50.0]}
            }
        ] + [dummy_action] * (env.max_actions_per_step - 1)
    }
    obs, reward, _, _, info = env.step(action)
    
    print(f"\nAction: Try to invest in firm 999 (doesn't exist)")
    action_result = info['action_results'][0]
    print(f"Result: {action_result['result']}")
    print(f"Total profit: ${info['total_profit']:.2f}")
    print(f"\nReward breakdown:")
    penalty = config.environment.invalid_firm_penalty
    profit_reward = info['total_profit'] * config.environment.profit_multiplier
    print(f"  Invalid firm penalty: ${penalty:>8.2f} (penalty for bad firm ID)")
    print(f"  Profit reward:        ${profit_reward:>8.2f} (profit × multiplier)")
    print(f"  Total reward:         ${reward:>8.2f}")
    
    # Test 4: Invalid investment amount
    print("\n" + "="*70)
    print("TEST 4: Invalid Investment Amount (Single Action)")
    print("="*70)
    action = {
        'num_actions': 1,
        'actions': [
            {
                'op': 0,
                'invest': {'firm_id': 0, 'amount': [10000000.0]},  # Way too high
                'create': {'sector': 0, 'initial_capital': [10000.0], 'location': [50.0, 50.0]}
            }
        ] + [dummy_action] * (env.max_actions_per_step - 1)
    }
    obs, reward, _, _, info = env.step(action)
    
    print(f"\nAction: Try to invest $10,000,000 (exceeds max)")
    action_result = info['action_results'][0]
    print(f"Result: {action_result['result']}")
    print(f"Total profit: ${info['total_profit']:.2f}")
    print(f"\nReward breakdown:")
    penalty = config.environment.invalid_action_penalty
    profit_reward = info['total_profit'] * config.environment.profit_multiplier
    print(f"  Invalid action penalty: ${penalty:>8.2f} (penalty for invalid amount)")
    print(f"  Profit reward:          ${profit_reward:>8.2f} (profit × multiplier)")
    print(f"  Total reward:           ${reward:>8.2f}")
    
    # Test 5: Multiple actions at once
    print("\n" + "="*70)
    print("TEST 5: Multiple Actions (New Feature!)")
    print("="*70)
    action = {
        'num_actions': 3,
        'actions': [
            {
                'op': 0,  # Invest in firm 0
                'invest': {'firm_id': 0, 'amount': [500.0]},
                'create': {'sector': 0, 'initial_capital': [10000.0], 'location': [50.0, 50.0]}
            },
            {
                'op': 0,  # Invest in firm 1
                'invest': {'firm_id': 1, 'amount': [1500.0]},
                'create': {'sector': 0, 'initial_capital': [10000.0], 'location': [50.0, 50.0]}
            },
            {
                'op': 1,  # Create new company
                'invest': {'firm_id': 0, 'amount': [0.0]},
                'create': {'sector': 2, 'initial_capital': [30000.0], 'location': [75.0, 25.0]}
            }
        ] + [dummy_action] * (env.max_actions_per_step - 3)
    }
    obs, reward, _, _, info = env.step(action)
    
    print(f"\nActions: Invest $500 in firm 0, $1,500 in firm 1, create new company")
    print(f"Number of actions: {info['num_actions']}")
    print(f"Valid actions: {info['num_valid_actions']}")
    print(f"Invalid actions: {info['num_invalid_actions']}")
    print(f"\nAction Results:")
    for i, result in enumerate(info['action_results']):
        print(f"  {i+1}. {result['result']}: investment=${result.get('investment_amount', 0):.2f}")
    print(f"\nNew firm count: {info['num_firms']}")
    print(f"Total profit: ${info['total_profit']:.2f}")
    print(f"Total reward: ${reward:>8.2f}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nAll reward calculations are now handled by _calculate_reward():")
    print("  ✓ Action-specific rewards (invest/create)")
    print("  ✓ Penalties for invalid actions")
    print("  ✓ Profit-based rewards")
    print("  ✓ Multi-action support (NEW!)")
    print("  ✓ Easy to extend with new reward components")
    print("\nInfo dict includes:")
    print("  - action_results: List of individual action results")
    print("  - num_actions: Number of actions taken")
    print("  - num_valid_actions: Number of successful actions")
    print("  - num_invalid_actions: Number of failed actions")
    print("  - investment_amount: Total investment across all actions")
    print("  - total_profit: Profit from all companies")
    print("  - total_logistic_cost: Logistics costs")
    print("  - num_firms: Current number of firms")
    print("  - num_deaths: Companies that died this step")
    print("\nMulti-Action System:")
    print(f"  - Max actions per step: {env.max_actions_per_step}")
    print("  - Agent chooses 0 to max_actions_per_step actions")
    print("  - Rewards accumulate across all actions")
    print("  - Each action validated independently")


if __name__ == "__main__":
    demo_reward_system()
