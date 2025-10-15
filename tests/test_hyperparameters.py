#!/usr/bin/env python3
"""
Test script to verify env.py respects all hyperparameters from config.

Tests that all parameters from the environment config are properly used.
"""

import os
import sys
import pytest
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import load_config, Config, EnvironmentConfig
from env import IndustryEnv


class TestHyperparameters:
    """Test suite for hyperparameter respect."""
    
    def test_spatial_parameters(self):
        """Test that spatial parameters are respected."""
        config = Config()
        config.environment.size = 250.0
        config.environment.num_sectors = 8
        config.environment.max_episode_steps = 500
        
        env = IndustryEnv(config.environment)
        
        assert env.size == 250.0, f"Size mismatch: {env.size} != 250.0"
        assert env.num_sectors == 8, f"Num sectors mismatch: {env.num_sectors} != 8"
        assert env.max_episode_steps == 500, f"Max episode steps mismatch: {env.max_episode_steps} != 500"
        
        print("✅ Spatial parameters respected")
    
    def test_capacity_parameters(self):
        """Test that capacity parameters are respected."""
        config = Config()
        config.environment.max_company = 500
        config.environment.initial_firms = 20
        
        env = IndustryEnv(config.environment)
        
        assert env.max_company == 500, f"Max company mismatch"
        
        # Test initial firms
        obs, info = env.reset(seed=42, options={"initial_firms": 20})
        assert obs['num_firms'] == 20, f"Initial firms mismatch: {obs['num_firms']} != 20"
        
        print("✅ Capacity parameters respected")
    
    def test_company_parameters(self):
        """Test that company parameters are respected."""
        config = Config()
        config.environment.op_cost_rate = 0.08
        config.environment.initial_capital_min = 20000.0
        config.environment.initial_capital_max = 80000.0
        config.environment.death_threshold = 5000.0
        
        env = IndustryEnv(config.environment)
        
        assert env.op_cost_rate == 0.08, "Op cost rate mismatch"
        assert env.initial_capital_min == 20000.0, "Initial capital min mismatch"
        assert env.initial_capital_max == 80000.0, "Initial capital max mismatch"
        assert env.death_threshold == 5000.0, "Death threshold mismatch"
        
        # Test that companies are created with capital in range
        obs, info = env.reset(seed=42, options={"initial_firms": 10})
        
        # Import sector_relations to check multipliers
        from env.sector import sector_relations
        
        for company in env.companies:
            assert 20000.0 <= company.capital <= 80000.0, \
                f"Capital {company.capital} out of range"
            # Company op_cost_rate includes sector multiplier
            sector_multiplier = sector_relations[company.sector_id].operating_cost_multiplier
            expected_op_cost_rate = 0.08 * sector_multiplier
            assert abs(company.op_cost_rate - expected_op_cost_rate) < 0.001, \
                f"Company op_cost_rate {company.op_cost_rate} != expected {expected_op_cost_rate} " \
                f"(base rate 0.08 × sector {company.sector_id} multiplier {sector_multiplier})"
        
        print("✅ Company parameters respected")
    
    def test_supply_chain_parameters(self):
        """Test that supply chain parameters are respected."""
        config = Config()
        config.environment.enable_supply_chain = True
        config.environment.trade_volume_fraction = 0.05
        config.environment.revenue_rate = 2.5
        config.environment.min_distance_epsilon = 0.5
        
        env = IndustryEnv(config.environment)
        
        assert env.enable_supply_chain == True, "Enable supply chain mismatch"
        assert env.trade_volume_fraction == 0.05, "Trade volume fraction mismatch"
        assert env.revenue_rate == 2.5, "Revenue rate mismatch"
        assert env.min_distance_epsilon == 0.5, "Min distance epsilon mismatch"
        
        # Test that companies get correct parameters
        obs, info = env.reset(seed=42, options={"initial_firms": 5})
        for company in env.companies:
            assert company.revenue_rate == 2.5, "Company revenue_rate not set"
            assert company.min_distance_epsilon == 0.5, "Company min_distance_epsilon not set"
        
        print("✅ Supply chain parameters respected")
    
    def test_logistics_parameters(self):
        """Test that logistics parameters are respected."""
        config = Config()
        config.environment.logistic_cost_rate = 300.0
        config.environment.disable_logistic_costs = False
        
        env = IndustryEnv(config.environment)
        
        assert env.logistic_cost_rate == 300.0, "Logistic cost rate mismatch"
        assert env.disable_logistic_costs == False, "Disable logistic costs mismatch"
        
        # Test that companies get correct logistic cost rate
        obs, info = env.reset(seed=42, options={"initial_firms": 3})
        for company in env.companies:
            assert company.logistic_cost_rate == 300.0, "Company logistic_cost_rate not set"
        
        # Test disabling logistic costs
        config.environment.disable_logistic_costs = True
        env2 = IndustryEnv(config.environment)
        assert env2.disable_logistic_costs == True, "Disable flag not set"
        
        print("✅ Logistics parameters respected")
    
    def test_reward_parameters(self):
        """Test that reward parameters are respected."""
        config = Config()
        config.environment.investment_multiplier = 0.05
        config.environment.creation_reward = 100.0
        config.environment.invalid_action_penalty = -50.0
        config.environment.invalid_firm_penalty = -25.0
        config.environment.profit_multiplier = 0.005
        
        env = IndustryEnv(config.environment)
        
        assert env.investment_multiplier == 0.05, "Investment multiplier mismatch"
        assert env.creation_reward == 100.0, "Creation reward mismatch"
        assert env.invalid_action_penalty == -50.0, "Invalid action penalty mismatch"
        assert env.invalid_firm_penalty == -25.0, "Invalid firm penalty mismatch"
        assert env.profit_multiplier == 0.005, "Profit multiplier mismatch"
        
        # Test that rewards are calculated correctly
        obs, info = env.reset(seed=42, options={"initial_firms": 5})
        
        # Test creation reward (using new multi-action format)
        from tests import create_single_action
        action_create = create_single_action(
            op=1,
            invest_dict={"firm_id": 0, "amount": [1000.0]},
            create_dict={"sector": 0, "initial_capital": [50000.0], "location": [50.0, 50.0]},
            max_actions=env.max_actions_per_step
        )
        obs, reward, _, _, info = env.step(action_create)
        # Reward should include creation_reward (100.0) plus profit component
        # The total reward will vary based on company profits, but should be positive
        # and include the creation reward as a major component
        assert reward > 0, \
            f"Creation should give positive reward, got: {reward}"
        # Check that the reward is in a reasonable range (creation_reward +/- profit adjustments)
        assert 0 < reward < 200.0, \
            f"Creation reward out of expected range: {reward}"
        
        print("✅ Reward parameters respected")
    
    def test_ablation_parameters(self):
        """Test that ablation parameters are respected."""
        config = Config()
        config.environment.disable_supply_chain = True
        config.environment.fixed_locations = True
        config.environment.allow_negative_capital = True
        
        env = IndustryEnv(config.environment)
        
        assert env.disable_supply_chain == True, "Disable supply chain mismatch"
        assert env.fixed_locations == True, "Fixed locations mismatch"
        assert env.allow_negative_capital == True, "Allow negative capital mismatch"
        
        # Test fixed locations
        obs, info = env.reset(seed=42, options={"initial_firms": 4})
        locations = [c.location for c in env.companies]
        # Check that locations are unique (grid pattern)
        assert len(set(locations)) == len(locations), "Fixed locations not unique"
        
        print("✅ Ablation parameters respected")
    
    def test_max_episode_steps(self):
        """Test that max_episode_steps is respected."""
        config = Config()
        config.environment.max_episode_steps = 10
        
        env = IndustryEnv(config.environment)
        
        obs, info = env.reset(seed=42, options={"initial_firms": 3})
        
        # Run exactly max_episode_steps
        from tests import create_single_action
        for i in range(10):
            action = create_single_action(
                op=0,
                invest_dict={"firm_id": 0, "amount": [100.0]},
                create_dict={"sector": 0, "initial_capital": [10000.0], "location": [50.0, 50.0]},
                max_actions=env.max_actions_per_step
            )
            obs, reward, terminated, truncated, info = env.step(action)
            
            if i < 9:
                assert not truncated, f"Episode truncated too early at step {i+1}"
            else:
                assert truncated, f"Episode not truncated at step {i+1}"
        
        print("✅ Max episode steps respected")
    
    def test_investment_constraints(self):
        """Test that investment min/max are respected."""
        config = Config()
        config.environment.investment_min = 100.0
        config.environment.investment_max = 50000.0
        
        env = IndustryEnv(config.environment)
        obs, info = env.reset(seed=42, options={"initial_firms": 5})
        
        # Test valid investment
        from tests import create_single_action
        action_valid = create_single_action(
            op=0,
            invest_dict={"firm_id": 0, "amount": [10000.0]},
            create_dict={"sector": 0, "initial_capital": [10000.0], "location": [50.0, 50.0]},
            max_actions=env.max_actions_per_step
        )
        obs, reward_valid, _, _, info = env.step(action_valid)
        # Should not be the invalid action penalty
        assert reward_valid != env.invalid_action_penalty, \
            "Valid investment should not be penalized"
        
        # Test invalid investment (too low) - should be rejected
        action_invalid_low = create_single_action(
            op=0,
            invest_dict={"firm_id": 0, "amount": [50.0]},  # Below minimum
            create_dict={"sector": 0, "initial_capital": [10000.0], "location": [50.0, 50.0]},
            max_actions=env.max_actions_per_step
        )
        obs, reward_low, _, _, info = env.step(action_invalid_low)
        # Note: The environment checks if amount is within range
        # If not, it applies invalid_action_penalty or invalid_firm_penalty
        # The actual behavior depends on implementation
        assert reward_low <= 0, \
            "Invalid investment (too low) should not give positive reward"
        
        # Test invalid investment (too high)
        action_invalid_high = create_single_action(
            op=0,
            invest_dict={"firm_id": 0, "amount": [100000.0]},  # Above maximum
            create_dict={"sector": 0, "initial_capital": [10000.0], "location": [50.0, 50.0]},
            max_actions=env.max_actions_per_step
        )
        obs, reward_high, _, _, info = env.step(action_invalid_high)
        assert reward_high <= 0, \
            "Invalid investment (too high) should not give positive reward"
        
        print("✅ Investment constraints respected")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("HYPERPARAMETER RESPECT TEST SUITE")
    print("="*70)
    
    test_suite = TestHyperparameters()
    
    try:
        test_suite.test_spatial_parameters()
        test_suite.test_capacity_parameters()
        test_suite.test_company_parameters()
        test_suite.test_supply_chain_parameters()
        test_suite.test_logistics_parameters()
        test_suite.test_reward_parameters()
        test_suite.test_ablation_parameters()
        test_suite.test_max_episode_steps()
        test_suite.test_investment_constraints()
        
        print("\n" + "="*70)
        print("✅ ALL HYPERPARAMETERS RESPECTED!")
        print("="*70)
        print("\nenv.py properly uses all configuration parameters:")
        print("  ✓ Spatial parameters (size, sectors, episode steps)")
        print("  ✓ Capacity parameters (max companies, initial firms)")
        print("  ✓ Company parameters (costs, capital ranges, death)")
        print("  ✓ Supply chain parameters (trade, revenue, epsilon)")
        print("  ✓ Logistics parameters (costs, disable flag)")
        print("  ✓ Reward parameters (multipliers, penalties)")
        print("  ✓ Ablation flags (disable flags, fixed locations)")
        print("  ✓ Episode management (max steps)")
        print("  ✓ Investment constraints (min/max)")
        print()
        return 0
        
    except AssertionError as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print("\n" + "="*70)
        print("❌ UNEXPECTED ERROR")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
