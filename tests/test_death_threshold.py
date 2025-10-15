#!/usr/bin/env python3
"""Test script for company death threshold mechanism."""

import os
import sys
import pytest
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from env import IndustryEnv


class TestDeathThreshold:
    """Test suite for death threshold mechanism."""
    
    def test_no_death_threshold(self):
        """Test that companies survive with negative capital when death_threshold = 0.0"""
        config = Config()
        config.environment.death_threshold = 0.0
        config.environment.allow_negative_capital = True
        
        env = IndustryEnv(config.environment)
        obs, _ = env.reset(seed=42, options={"initial_firms": 10})
        
        initial_firms = env.num_firms
        assert initial_firms == 10, "Should start with 10 firms"
        
        # Run simulation with random actions
        deaths = 0
        for i in range(20):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            deaths += info.get('num_deaths', 0)
        
        print(f"✅ No death threshold test: {deaths} deaths (companies can have negative capital)")
        # With death_threshold=0 and allow_negative_capital=True, deaths should be 0
        assert deaths == 0, "No companies should die with death_threshold=0 and allow_negative_capital=True"
    
    def test_positive_death_threshold(self):
        """Test that companies die when capital falls below positive threshold."""
        config = Config()
        config.environment.death_threshold = 5000.0
        config.environment.allow_negative_capital = False
        config.environment.max_episode_steps = 50
        
        env = IndustryEnv(config.environment)
        obs, _ = env.reset(seed=42, options={"initial_firms": 10})
        
        initial_firms = env.num_firms
        assert initial_firms == 10, "Should start with 10 firms"
        
        # Show initial capital distribution
        capitals = [c.capital for c in env.companies]
        print(f"Initial capital range: ${min(capitals):,.2f} - ${max(capitals):,.2f}")
        
        # Run simulation
        deaths = 0
        for i in range(20):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            deaths += info.get('num_deaths', 0)
        
        print(f"✅ Positive death threshold test: {deaths} deaths with threshold=$5,000")
        
        # Verify remaining companies are above threshold
        if env.companies:
            for company in env.companies:
                assert company.capital >= 5000.0, \
                    f"Company has capital ${company.capital} below threshold $5,000"
    
    def test_high_death_threshold(self):
        """Test aggressive culling with high death threshold."""
        config = Config()
        config.environment.death_threshold = 50000.0
        config.environment.allow_negative_capital = False
        config.environment.initial_capital_min = 10000.0
        config.environment.initial_capital_max = 100000.0
        
        env = IndustryEnv(config.environment)
        obs, _ = env.reset(seed=42, options={"initial_firms": 10})
        
        initial_firms = env.num_firms
        
        # Show initial capital distribution
        capitals = [c.capital for c in env.companies]
        print(f"Initial capital range: ${min(capitals):,.2f} - ${max(capitals):,.2f}")
        below_threshold = sum(1 for c in capitals if c < 50000.0)
        print(f"Companies below threshold at start: {below_threshold}")
        
        # Run simulation
        deaths = 0
        for i in range(20):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            deaths += info.get('num_deaths', 0)
        
        final_firms = env.num_firms
        print(f"✅ High death threshold test: {deaths} deaths, {initial_firms} → {final_firms} firms")
        
        # Verify remaining companies are above threshold
        if env.companies:
            for company in env.companies:
                assert company.capital >= 50000.0, \
                    f"Company has capital ${company.capital} below threshold $50,000"
    
    def test_allow_negative_capital_flag(self):
        """Test that allow_negative_capital flag prevents death."""
        config = Config()
        config.environment.death_threshold = 10000.0
        config.environment.allow_negative_capital = True
        
        env = IndustryEnv(config.environment)
        obs, _ = env.reset(seed=42, options={"initial_firms": 5})
        
        # Manually set a company's capital very low
        env.companies[0].capital = -5000.0
        
        # Take a step
        from tests import create_single_action
        action = create_single_action(
            op=0,
            invest_dict={"firm_id": 1, "amount": [100.0]},
            create_dict={"initial_capital": [10000.0], "sector": 0, "location": [50.0, 50.0]},
            max_actions=env.max_actions_per_step
        )
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Company should still exist despite negative capital
        num_deaths = info.get('num_deaths', 0)
        assert num_deaths == 0, "No companies should die when allow_negative_capital=True"
        print(f"✅ Allow negative capital test: Company with ${env.companies[0].capital} survived")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("COMPANY DEATH THRESHOLD TEST SUITE")
    print("="*70)
    
    test_suite = TestDeathThreshold()
    
    try:
        test_suite.test_no_death_threshold()
        test_suite.test_positive_death_threshold()
        test_suite.test_high_death_threshold()
        test_suite.test_allow_negative_capital_flag()
        
        print("\n" + "="*70)
        print("✅ ALL DEATH THRESHOLD TESTS PASSED")
        print("="*70)
        print("\nDeath threshold mechanism working correctly:")
        print("  ✓ death_threshold = 0: No bankruptcy")
        print("  ✓ death_threshold > 0: Companies die below threshold")
        print("  ✓ allow_negative_capital flag: Prevents death")
        print()
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
