#!/usr/bin/env python3
"""
Test script to verify config.py integration with env.py.

This script tests:
1. Config loading from config.json
2. Environment creation with config parameters
3. Environment respects all hyperparameters
"""

import os
import sys
import tempfile
import json
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import load_config, Config, EnvironmentConfig
from env import IndustryEnv


class TestConfigIntegration:
    """Test suite for config integration."""
    
    def test_config_loading(self):
        """Test that config.py can load config.json."""
        # Load default config
        config = load_config("config.json")
        
        # Verify key parameters exist
        assert hasattr(config, 'environment'), "Missing environment config"
        assert hasattr(config, 'training'), "Missing training config"
        assert isinstance(config.environment, EnvironmentConfig), "environment should be EnvironmentConfig"
        
        # Verify environment parameters
        assert config.environment.size == 100.0, "Environment size mismatch"
        assert config.environment.max_company == 1000, "Max company mismatch"
        assert config.environment.logistic_cost_rate == 100.0, "Logistic cost rate mismatch"
        assert config.environment.revenue_rate == 1.0, "Revenue rate mismatch"
        assert config.environment.death_threshold == 0.0, "Death threshold mismatch"
        
        # Verify product system parameters
        assert hasattr(config.environment, 'enable_products'), "Missing enable_products parameter"
        assert hasattr(config.environment, 'production_capacity_ratio'), "Missing production_capacity_ratio parameter"
        assert hasattr(config.environment, 'purchase_budget_ratio'), "Missing purchase_budget_ratio parameter"
        assert config.environment.enable_products == True, "Product system should be enabled by default"
        assert config.environment.production_capacity_ratio == 0.1, "Production capacity ratio mismatch"
        assert config.environment.purchase_budget_ratio == 0.2, "Purchase budget ratio mismatch"
        
        print("✅ Config loaded successfully (including product parameters)")
    
    def test_env_creation_with_config(self):
        """Test that environment can be created with config parameters."""
        config = load_config("config.json")
        
        # Create environment with config object
        env = IndustryEnv(config.environment)
        
        # Verify environment parameters
        assert env.size == config.environment.size, "Environment size not set correctly"
        assert env.max_company == config.environment.max_company, "Max company not set correctly"
        assert env.num_sectors == config.environment.num_sectors, "Num sectors not set correctly"
        assert env.logistic_cost_rate == config.environment.logistic_cost_rate, "Logistic cost rate not set correctly"
        assert env.revenue_rate == config.environment.revenue_rate, "Revenue rate not set correctly"
        assert env.death_threshold == config.environment.death_threshold, "Death threshold not set correctly"
        
        # Verify product system parameters
        assert env.enable_products == config.environment.enable_products, "Enable products not set correctly"
        assert env.production_capacity_ratio == config.environment.production_capacity_ratio, "Production capacity ratio not set correctly"
        assert env.purchase_budget_ratio == config.environment.purchase_budget_ratio, "Purchase budget ratio not set correctly"
        
        print("✅ Environment created successfully with config parameters (including products)")
    
    def test_env_respects_hyperparameters(self):
        """Test that environment actually uses the hyperparameters during operation."""
        # Create environment with custom parameters
        custom_revenue_rate = 2.5
        custom_death_threshold = 5000.0
        custom_logistic_cost_rate = 200.0
        
        config = Config()
        config.environment.revenue_rate = custom_revenue_rate
        config.environment.death_threshold = custom_death_threshold
        config.environment.logistic_cost_rate = custom_logistic_cost_rate
        
        env = IndustryEnv(config.environment)
        
        # Initialize environment and create companies
        obs, info = env.reset(seed=42, options={"initial_firms": 5})
        
        # Verify companies have correct parameters
        for company in env.companies:
            assert company.revenue_rate == custom_revenue_rate, \
                f"Company revenue rate mismatch: {company.revenue_rate} != {custom_revenue_rate}"
            assert company.logistic_cost_rate == custom_logistic_cost_rate, \
                f"Company logistic cost rate mismatch: {company.logistic_cost_rate} != {custom_logistic_cost_rate}"
            # Verify product system parameters
            assert hasattr(company, 'production_capacity_ratio'), "Company missing production_capacity_ratio"
            assert hasattr(company, 'purchase_budget_ratio'), "Company missing purchase_budget_ratio"
            assert hasattr(company, 'product_inventory'), "Company missing product_inventory"
            assert hasattr(company, 'tier'), "Company missing tier"
        
        print("✅ Companies created with correct hyperparameters (including product system)")
        
        # Test death mechanism
        initial_num_firms = len(env.companies)
        
        # Manually set one company's capital below death threshold
        if env.companies and custom_death_threshold > 0:
            # Set capital very low to ensure it stays below threshold even after step
            # Account for potential revenue from supply chain
            env.companies[0].capital = 1.0  # Very low capital, will definitely die
            
            # Take a step - this should trigger death check
            from tests import create_single_action
            action = create_single_action(
                op=0,  # invest
                invest_dict={"firm_id": 1 if len(env.companies) > 1 else 0, "amount": [1000.0]},
                create_dict={"initial_capital": [10000.0], "sector": 0, "location": [50.0, 50.0]},
                max_actions=env.max_actions_per_step
            )
            obs, reward, terminated, truncated, info = env.step(action)
            
            final_num_firms = len(env.companies)
            assert final_num_firms < initial_num_firms, "Death mechanism did not remove company"
            print(f"✅ Death mechanism working: {initial_num_firms} → {final_num_firms} firms")
    
    def test_custom_config_file(self):
        """Test loading from a custom config file with different values."""
        # Create temporary config file with custom values
        custom_config = {
            "environment": {
                "size": 200.0,
                "max_company": 500,
                "num_sectors": 10,
                "initial_firms": 10,
                "max_episode_steps": 1000,
                "op_cost_rate": 0.08,
                "initial_capital_min": 15000.0,
                "initial_capital_max": 50000.0,
                "investment_min": 0.0,
                "investment_max": 1000000.0,
                "new_company_capital_min": 1000.0,
                "new_company_capital_max": 1000000.0,
                "death_threshold": 8000.0,
                "trade_volume_fraction": 0.03,
                "revenue_rate": 1.5,
                "enable_supply_chain": True,
                "min_distance_epsilon": 0.1,
                "logistic_cost_rate": 150.0,
                "investment_multiplier": 0.01,
                "creation_reward": 50.0,
                "invalid_action_penalty": -20.0,
                "invalid_firm_penalty": -10.0,
                "profit_multiplier": 0.001,
                "figsize_width": 12,
                "figsize_height": 8,
                "dpi": 150,
                "show_plots": False,
                "save_plots": True,
                "plot_format": "png",
                "disable_logistic_costs": False,
                "disable_supply_chain": False,
                "fixed_locations": False,
                "allow_negative_capital": False,
                "enable_products": True,
                "production_capacity_ratio": 0.1,
                "purchase_budget_ratio": 0.2
            },
            "training": {
                "framework": "sb3",
                "algorithm": "ppo",
                "total_timesteps": 50000,
                "num_envs": 2,
                "learning_rate": 0.001,
                "batch_size": 128,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "n_steps": 1024,
                "n_epochs": 10,
                "seed": 123,
                "verbose": 1,
                "log_dir": "./logs",
                "checkpoint_dir": "./checkpoints",
                "save_freq": 5000,
                "eval_freq": 2500,
                "visualize_freq": 10000,
                "n_eval_episodes": 10,
                "save_best_model": True,
                "normalize_observations": False,
                "normalize_rewards": False,
                "clip_observations": 10.0,
                "clip_rewards": 10.0,
                "frame_stack": 1
            }
        }
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(custom_config, f, indent=2)
            temp_config_path = f.name
        
        try:
            # Load custom config
            config = load_config(temp_config_path)
            
            # Verify custom values
            assert config.environment.size == 200.0, "Custom size not loaded"
            assert config.environment.max_company == 500, "Custom max_company not loaded"
            assert config.environment.logistic_cost_rate == 150.0, "Custom logistic_cost_rate not loaded"
            assert config.environment.revenue_rate == 1.5, "Custom revenue_rate not loaded"
            assert config.environment.death_threshold == 8000.0, "Custom death_threshold not loaded"
            
            print("✅ Custom config loaded successfully")
            
            # Create environment with custom config
            env = IndustryEnv(config.environment)
            assert env.size == 200.0, "Environment not created with custom size"
            
            print("✅ Environment created with custom config")
            
        finally:
            # Clean up
            os.unlink(temp_config_path)


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("CONFIG INTEGRATION TEST SUITE")
    print("="*70)
    
    test_suite = TestConfigIntegration()
    
    try:
        test_suite.test_config_loading()
        test_suite.test_env_creation_with_config()
        test_suite.test_env_respects_hyperparameters()
        test_suite.test_custom_config_file()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        return 0
        
    except AssertionError as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"Error: {e}")
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
