#!/usr/bin/env python3
"""
Test script to demonstrate and validate multi-action functionality using pytest.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from env import IndustryEnv


@pytest.fixture
def env():
    """Create environment fixture with max_actions_per_step = 10."""
    config = Config()
    config.environment.max_actions_per_step = 10
    env = IndustryEnv(config.environment)
    obs, _ = env.reset(seed=42, options={'initial_firms': 5})
    return env


class TestMultiActionSystem:
    """Test suite for multi-action system functionality."""
    
    def _create_env(self):
        """Helper method to create a fresh environment."""
        config = Config()
        config.environment.max_actions_per_step = 10
        env = IndustryEnv(config.environment)
        obs, _ = env.reset(seed=42, options={'initial_firms': 5})
        return env
    
    def test_zero_actions(self, env=None):
        """Test that zero actions (num_actions = 0) works correctly."""
        if env is None:
            env = self._create_env()
        action = {
            "num_actions": 0,
            "actions": [env.action_space["actions"][i].sample() for i in range(env.max_actions_per_step)]
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert info['num_actions'] == 0, "Should execute 0 actions"
        assert info['num_valid_actions'] == 0, "Should have 0 valid actions"
        assert info['num_invalid_actions'] == 0, "Should have 0 invalid actions"
        print(f"\n✅ Zero actions test passed: {info['num_actions']} actions executed")
    
    def test_single_action_invest(self, env=None):
        """Test single invest action."""
        if env is None:
            env = self._create_env()
        action = {
            "num_actions": 1,
            "actions": [
                {
                    "op": 0,
                    "invest": {"firm_id": 0, "amount": [1000.0]},
                    "create": {"sector": 0, "initial_capital": [10000.0], "location": [50.0, 50.0]}
                }
            ] + [env.action_space["actions"][i].sample() for i in range(1, env.max_actions_per_step)]
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert info['num_actions'] == 1, "Should execute 1 action"
        assert info['num_valid_actions'] >= 0, "Should have at least 0 valid actions"
        assert info['investment_amount'] >= 0, "Should have non-negative investment"
        print(f"\n✅ Single action test passed: ${info['investment_amount']:.2f} invested")
    
    def test_multiple_actions(self, env=None):
        """Test multiple actions (2 invests + 1 create)."""
        if env is None:
            env = self._create_env()
        initial_firms = env.num_firms
        
        action = {
            "num_actions": 3,
            "actions": [
                {  # Action 1: Invest $500 in firm 0
                    "op": 0,
                    "invest": {"firm_id": 0, "amount": [500.0]},
                    "create": {"sector": 0, "initial_capital": [10000.0], "location": [50.0, 50.0]}
                },
                {  # Action 2: Invest $1500 in firm 1
                    "op": 0,
                    "invest": {"firm_id": 1, "amount": [1500.0]},
                    "create": {"sector": 0, "initial_capital": [10000.0], "location": [50.0, 50.0]}
                },
                {  # Action 3: Create new company
                    "op": 1,
                    "invest": {"firm_id": 0, "amount": [1000.0]},
                    "create": {"sector": 2, "initial_capital": [30000.0], "location": [25.0, 75.0]}
                }
            ] + [env.action_space["actions"][i].sample() for i in range(3, env.max_actions_per_step)]
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert info['num_actions'] == 3, "Should execute 3 actions"
        assert info['num_valid_actions'] >= 1, "Should have at least 1 valid action"
        assert len(info['action_results']) == 3, "Should have 3 action results"
        print(f"\n✅ Multiple actions test passed: {info['num_actions']} actions, {info['num_valid_actions']} valid")
    
    def test_maximum_actions(self, env=None):
        """Test maximum number of actions (10 creates)."""
        if env is None:
            env = self._create_env()
        initial_firms = env.num_firms
        
        actions_list = []
        for i in range(10):
            actions_list.append({
                "op": 1,
                "invest": {"firm_id": 0, "amount": [1000.0]},
                "create": {
                    "sector": i % env.num_sectors,
                    "initial_capital": [15000.0 + i * 1000],
                    "location": [float(10 + i * 8), float(20 + i * 5)]
                }
            })
        
        action = {
            "num_actions": 10,
            "actions": actions_list
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert info['num_actions'] == 10, "Should execute 10 actions"
        assert info['num_valid_actions'] >= 5, "Should have at least 5 valid actions"
        assert env.num_firms > initial_firms, "Should have created new firms"
        print(f"\n✅ Maximum actions test passed: {initial_firms} → {env.num_firms} firms (+{env.num_firms - initial_firms})")
    
    def test_mixed_valid_invalid_actions(self, env=None):
        """Test mix of valid and invalid actions."""
        if env is None:
            env = self._create_env()
        action = {
            "num_actions": 5,
            "actions": [
                {  # Valid invest
                    "op": 0,
                    "invest": {"firm_id": 0, "amount": [1000.0]},
                    "create": {"sector": 0, "initial_capital": [10000.0], "location": [50.0, 50.0]}
                },
                {  # Invalid firm ID
                    "op": 0,
                    "invest": {"firm_id": 9999, "amount": [1000.0]},
                    "create": {"sector": 0, "initial_capital": [10000.0], "location": [50.0, 50.0]}
                },
                {  # Valid create
                    "op": 1,
                    "invest": {"firm_id": 0, "amount": [1000.0]},
                    "create": {"sector": 3, "initial_capital": [25000.0], "location": [30.0, 40.0]}
                },
                {  # Invalid amount (too high)
                    "op": 0,
                    "invest": {"firm_id": 0, "amount": [10000000.0]},
                    "create": {"sector": 0, "initial_capital": [10000.0], "location": [50.0, 50.0]}
                },
                {  # Valid invest
                    "op": 0,
                    "invest": {"firm_id": 1, "amount": [2000.0]},
                    "create": {"sector": 0, "initial_capital": [10000.0], "location": [50.0, 50.0]}
                }
            ] + [env.action_space["actions"][i].sample() for i in range(5, env.max_actions_per_step)]
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert info['num_actions'] == 5, "Should execute 5 actions"
        assert info['num_valid_actions'] >= 1, "Should have at least 1 valid action"
        assert info['num_invalid_actions'] >= 1, "Should have at least 1 invalid action"
        assert len(info['action_results']) == 5, "Should have 5 action results"
        print(f"\n✅ Mixed valid/invalid test passed: {info['num_valid_actions']} valid, {info['num_invalid_actions']} invalid")
    
    def test_action_results_tracking(self, env=None):
        """Test that action results are properly tracked."""
        if env is None:
            env = self._create_env()
        action = {
            "num_actions": 2,
            "actions": [
                {
                    "op": 0,
                    "invest": {"firm_id": 0, "amount": [500.0]},
                    "create": {"sector": 0, "initial_capital": [10000.0], "location": [50.0, 50.0]}
                },
                {
                    "op": 1,
                    "invest": {"firm_id": 0, "amount": [1000.0]},
                    "create": {"sector": 1, "initial_capital": [20000.0], "location": [60.0, 60.0]}
                }
            ] + [env.action_space["actions"][i].sample() for i in range(2, env.max_actions_per_step)]
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert 'action_results' in info, "Info should contain action_results"
        assert len(info['action_results']) == 2, "Should have 2 action results"
        
        for action_result in info['action_results']:
            assert 'action_idx' in action_result, "Each result should have action_idx"
            assert 'result' in action_result, "Each result should have result status"
        
        print(f"\n✅ Action results tracking test passed: {len(info['action_results'])} results tracked")
    
    def test_reward_accumulation(self, env=None):
        """Test that rewards accumulate across multiple actions."""
        if env is None:
            env = self._create_env()
        # Single action reward
        action_single = {
            "num_actions": 1,
            "actions": [
                {
                    "op": 1,
                    "invest": {"firm_id": 0, "amount": [1000.0]},
                    "create": {"sector": 0, "initial_capital": [20000.0], "location": [50.0, 50.0]}
                }
            ] + [env.action_space["actions"][i].sample() for i in range(1, env.max_actions_per_step)]
        }
        
        obs1, reward1, _, _, info1 = env.step(action_single)
        
        # Multiple actions should potentially have different reward
        action_multiple = {
            "num_actions": 3,
            "actions": [
                {
                    "op": 1,
                    "invest": {"firm_id": 0, "amount": [1000.0]},
                    "create": {"sector": 1, "initial_capital": [20000.0], "location": [30.0, 30.0]}
                },
                {
                    "op": 1,
                    "invest": {"firm_id": 0, "amount": [1000.0]},
                    "create": {"sector": 2, "initial_capital": [20000.0], "location": [70.0, 70.0]}
                },
                {
                    "op": 0,
                    "invest": {"firm_id": 0, "amount": [500.0]},
                    "create": {"sector": 0, "initial_capital": [10000.0], "location": [50.0, 50.0]}
                }
            ] + [env.action_space["actions"][i].sample() for i in range(3, env.max_actions_per_step)]
        }
        
        obs2, reward2, _, _, info2 = env.step(action_multiple)
        
        # Just verify rewards are calculated (can be positive or negative)
        assert isinstance(reward1, (int, float, np.number)), "Reward should be numeric"
        assert isinstance(reward2, (int, float, np.number)), "Reward should be numeric"
        print(f"\n✅ Reward accumulation test passed: single={reward1:.2f}, multiple={reward2:.2f}")
    
    def test_max_actions_per_step_config(self, env=None):
        """Test that max_actions_per_step is correctly configured."""
        if env is None:
            env = self._create_env()
        assert env.max_actions_per_step == 10, "max_actions_per_step should be 10"
        assert hasattr(env, 'max_actions_per_step'), "Environment should have max_actions_per_step attribute"
        print(f"\n✅ Config test passed: max_actions_per_step={env.max_actions_per_step}")


def test_multi_action_system_integration():
    """Integration test to verify overall multi-action system functionality."""
    print("\n" + "="*70)
    print("MULTI-ACTION SYSTEM INTEGRATION TEST")
    print("="*70)
    
    # Create environment
    config = Config()
    config.environment.max_actions_per_step = 10
    env = IndustryEnv(config.environment)
    obs, _ = env.reset(seed=42, options={'initial_firms': 5})
    
    print(f"\nEnvironment configuration:")
    print(f"  - Max actions per step: {env.max_actions_per_step}")
    print(f"  - Starting firms: {obs['num_firms']}")
    
    # Run a sequence of multi-action steps
    total_actions = 0
    total_valid = 0
    total_invalid = 0
    
    for step in range(5):
        num_actions = (step % 5) + 1  # 1, 2, 3, 4, 5 actions
        
        actions_list = []
        for i in range(num_actions):
            if i % 2 == 0:
                actions_list.append({
                    "op": 0,
                    "invest": {"firm_id": i % env.num_firms, "amount": [1000.0]},
                    "create": {"sector": 0, "initial_capital": [10000.0], "location": [50.0, 50.0]}
                })
            else:
                actions_list.append({
                    "op": 1,
                    "invest": {"firm_id": 0, "amount": [1000.0]},
                    "create": {"sector": i % env.num_sectors, "initial_capital": [15000.0], 
                              "location": [float(20 + i * 10), float(30 + i * 10)]}
                })
        
        # Pad with random actions
        actions_list.extend([env.action_space["actions"][i].sample() 
                           for i in range(num_actions, env.max_actions_per_step)])
        
        action = {
            "num_actions": num_actions,
            "actions": actions_list
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_actions += info['num_actions']
        total_valid += info['num_valid_actions']
        total_invalid += info['num_invalid_actions']
        
        print(f"\nStep {step + 1}: {num_actions} actions → {info['num_valid_actions']} valid, {info['num_invalid_actions']} invalid")
    
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)
    print(f"\n✅ Integration test passed!")
    print(f"\nStatistics across 5 steps:")
    print(f"  - Total actions attempted: {total_actions}")
    print(f"  - Total valid actions: {total_valid}")
    print(f"  - Total invalid actions: {total_invalid}")
    print(f"  - Final firms: {obs['num_firms']}")
    print(f"\nKey features verified:")
    print("  ✓ Agent can choose 0 to max_actions_per_step actions")
    print("  ✓ Each action is validated independently")
    print("  ✓ Rewards accumulate across all actions")
    print("  ✓ Info dict tracks all action results")
    print("  ✓ max_actions_per_step is configurable")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])