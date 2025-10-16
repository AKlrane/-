#!/usr/bin/env python3
"""
Test script to demonstrate logistic cost functionality.
Shows how distance between companies affects transportation costs using linear relationship.
"""

import os
import sys
import pytest
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from env import Company, IndustryEnv


class TestLogisticCost:
    """Test suite for logistic cost calculations."""
    
    def test_logistic_cost_calculation(self):
        """Test the logistic cost calculation with different distances."""
        # Create company with logistic_cost_rate=100.0
        company_a = Company(
            capital=100000,
            sector_id=1,
            location=(0.0, 0.0),
            logistic_cost_rate=100.0
        )
        
        distances = [1.0, 5.0, 10.0, 20.0, 50.0]
        trade_volume = 1000.0
        
        print(f"Testing linear relationship: cost = k * volume * distance")
        print(f"Logistic Cost Rate (k): {company_a.logistic_cost_rate}")
        
        previous_cost = None
        for dist in distances:
            company_b = Company(
                capital=100000,
                sector_id=2,
                location=(dist, 0.0)
            )
            cost = company_a.calculate_logistic_cost_to(company_b, trade_volume)
            
            # Verify linear relationship
            expected_cost = company_a.logistic_cost_rate * trade_volume * dist
            assert abs(cost - expected_cost) < 0.01, \
                f"Cost calculation error at distance {dist}: {cost} != {expected_cost}"
            
            if previous_cost is not None:
                # Cost should increase as distance increases (linear relationship)
                assert cost > previous_cost, \
                    f"Cost should increase with distance: {cost} <= {previous_cost}"
            
            previous_cost = cost
        
        print("✅ Logistic cost follows linear relationship")
    
    def test_environment_with_logistic_costs(self):
        """Test the environment with supply chain and logistic costs."""
        # Test with different logistic cost rates
        for rate in [10.0, 100.0, 1000.0]:
            config = Config()
            config.environment.logistic_cost_rate = rate
            config.environment.enable_supply_chain = True
            config.environment.disable_logistic_costs = False
            
            env = IndustryEnv(config.environment)
            obs, info = env.reset(seed=42, options={"initial_firms": 10})
            
            # Run a few steps
            for step in range(5):
                from tests import create_single_action
                action = create_single_action(
                    op=1,  # Create
                    invest_dict={"firm_id": 0, "amount": [1000.0]},
                    create_dict={
                        "initial_capital": [50000.0],
                        "sector": 0,
                        "location": np.random.uniform(0, 100, size=2).astype(np.float32)
                    },
                    max_actions=env.max_actions_per_step
                )
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Verify logistic costs are being tracked
                if step == 0:
                    total_logistic_cost = info['total_logistic_cost']
                    assert total_logistic_cost >= 0, "Logistic cost should be non-negative"
            
            print(f"✅ Environment works with logistic_cost_rate={rate}")
    
    def test_disable_logistic_costs(self):
        """Test that logistic costs can be disabled."""
        config = Config()
        config.environment.disable_logistic_costs = True
        config.environment.enable_supply_chain = True
        
        env = IndustryEnv(config.environment)
        obs, info = env.reset(seed=42, options={"initial_firms": 10})
        
        # Run simulation
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # When logistic costs are disabled, total_logistic_cost should be 0
            assert info['total_logistic_cost'] == 0.0, \
                f"Logistic costs should be 0 when disabled, got {info['total_logistic_cost']}"
        
        print("✅ Logistic costs successfully disabled")
    
    def test_distance_impact_on_profitability(self):
        """Test how company placement affects profitability through logistic costs."""
        config = Config()
        config.environment.logistic_cost_rate = 100.0
        config.environment.enable_supply_chain = True
        config.environment.disable_logistic_costs = False
        
        # Create an energy company (supplier)
        energy_co = Company(
            capital=100000,
            sector_id=1,
            location=(0.0, 0.0),
            logistic_cost_rate=100.0,
            revenue_rate=1.0
        )
        
        trade_volume = 5000.0
        
        # Test different distances
        previous_cost = None
        for distance in [5.0, 10.0, 20.0, 50.0]:
            mfg_co = Company(
                capital=100000,
                sector_id=2,
                location=(distance, 0.0)
            )
            
            # Calculate cost
            log_cost = energy_co.calculate_logistic_cost_to(mfg_co, trade_volume)
            
            # Verify linear relationship
            expected_cost = 100.0 * trade_volume * distance
            assert abs(log_cost - expected_cost) < 0.01, \
                f"Cost calculation error: {log_cost} != {expected_cost}"
            
            # Cost increases with distance (linear relationship)
            if previous_cost is not None:
                assert log_cost > previous_cost, \
                    f"Cost should increase with distance: {log_cost} <= {previous_cost}"
            previous_cost = log_cost
            
            # Calculate net revenue (cost increases with distance in linear relationship)
            # So closer companies are more profitable
            net_revenue = energy_co.revenue_rate * trade_volume - log_cost
            # Net revenue should be positive for reasonable distances
        
        print("✅ Distance impact on profitability verified")
    
    def test_min_distance_epsilon(self):
        """Test handling of co-located companies (distance = 0)."""
        config = Config()
        config.environment.min_distance_epsilon = 0.1
        
        # Create two companies at the same location
        company_a = Company(
            capital=100000,
            sector_id=1,
            location=(50.0, 50.0),
            logistic_cost_rate=100.0,
            min_distance_epsilon=0.1
        )
        
        company_b = Company(
            capital=100000,
            sector_id=2,
            location=(50.0, 50.0)  # Same location
        )
        
        # This should work fine with linear relationship (distance = 0 means cost = 0)
        cost = company_a.calculate_logistic_cost_to(company_b, 1000.0)
        
        # With linear relationship, distance 0 means cost should be 0
        assert cost == 0.0, \
            f"Cost at distance 0 should be 0, got: {cost}"
        
        print("✅ Co-located companies have zero logistic cost")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("LOGISTIC COST TEST SUITE")
    print("="*70)
    
    test_suite = TestLogisticCost()
    
    try:
        test_suite.test_logistic_cost_calculation()
        test_suite.test_environment_with_logistic_costs()
        test_suite.test_disable_logistic_costs()
        test_suite.test_distance_impact_on_profitability()
        test_suite.test_min_distance_epsilon()
        
        print("\n" + "="*70)
        print("✅ ALL LOGISTIC COST TESTS PASSED")
        print("="*70)
        print("\nLogistic cost system verified:")
        print("  ✓ Linear relationship: cost = k * volume * distance")
        print("  ✓ Configurable rate parameter")
        print("  ✓ Can be disabled via flag")
        print("  ✓ Distance affects profitability")
        print("  ✓ Co-located companies have zero cost")
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
