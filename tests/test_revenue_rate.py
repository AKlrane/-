#!/usr/bin/env python3
"""Test script for revenue_rate hyperparameter."""

import os
import sys
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from env import IndustryEnv, Company


class TestRevenueRate:
    """Test suite for revenue_rate parameter."""
    
    def test_default_revenue_rate(self):
        """Test that revenue_rate = 1.0 gives 1:1 order to revenue."""
        company = Company(
            capital=10000.0,
            sector_id=0,
            location=(50.0, 50.0),
            revenue_rate=1.0
        )
        company.add_revenue(100.0)
        
        assert company.revenue == 100.0, \
            f"Expected revenue=100.0, got {company.revenue}"
        assert company.orders == 1, \
            f"Expected orders=1, got {company.orders}"
        
        print("✅ Default revenue_rate = 1.0 works correctly")
    
    def test_doubled_revenue_rate(self):
        """Test that revenue_rate = 2.0 doubles revenue."""
        company = Company(
            capital=10000.0,
            sector_id=0,
            location=(50.0, 50.0),
            revenue_rate=2.0
        )
        company.add_revenue(100.0)
        
        assert company.revenue == 200.0, \
            f"Expected revenue=200.0, got {company.revenue}"
        assert company.orders == 1, \
            f"Expected orders=1, got {company.orders}"
        
        print("✅ revenue_rate = 2.0 doubles revenue")
    
    def test_halved_revenue_rate(self):
        """Test that revenue_rate = 0.5 halves revenue."""
        company = Company(
            capital=10000.0,
            sector_id=0,
            location=(50.0, 50.0),
            revenue_rate=0.5
        )
        company.add_revenue(100.0)
        
        assert company.revenue == 50.0, \
            f"Expected revenue=50.0, got {company.revenue}"
        assert company.orders == 1, \
            f"Expected orders=1, got {company.orders}"
        
        print("✅ revenue_rate = 0.5 halves revenue")
    
    def test_multiple_orders_accumulate(self):
        """Test that multiple orders accumulate correctly with revenue_rate."""
        company = Company(
            capital=10000.0,
            sector_id=0,
            location=(50.0, 50.0),
            revenue_rate=1.5
        )
        
        company.add_revenue(100.0)
        company.add_revenue(50.0)
        company.add_revenue(200.0)
        
        expected_revenue = (100.0 + 50.0 + 200.0) * 1.5
        assert company.revenue == expected_revenue, \
            f"Expected revenue={expected_revenue}, got {company.revenue}"
        assert company.orders == 3, \
            f"Expected orders=3, got {company.orders}"
        
        print("✅ Multiple orders accumulate correctly")
    
    def test_revenue_rate_in_environment(self):
        """Test that revenue_rate is properly used in environment."""
        config = Config()
        config.environment.revenue_rate = 2.5
        config.environment.enable_supply_chain = True
        config.environment.disable_supply_chain = False
        
        env = IndustryEnv(config.environment)
        obs, info = env.reset(seed=42, options={"initial_firms": 10})
        
        # Verify all companies have correct revenue_rate
        for company in env.companies:
            assert company.revenue_rate == 2.5, \
                f"Company revenue_rate mismatch: {company.revenue_rate} != 2.5"
        
        # Run a step to trigger supply chain
        from tests import create_single_action
        action = create_single_action(
            op=0,
            invest_dict={"firm_id": 0, "amount": [100.0]},
            create_dict={"sector": 0, "initial_capital": [10000.0], "location": [50.0, 50.0]},
            max_actions=env.max_actions_per_step
        )
        obs, reward, terminated, truncated, info = env.step(action)
        
        print("✅ revenue_rate properly used in environment")
    
    def test_revenue_calculation_formula(self):
        """Test the revenue calculation formula: revenue = revenue_rate × order_amount."""
        test_cases = [
            (1.0, 100.0, 100.0),
            (2.0, 100.0, 200.0),
            (0.5, 100.0, 50.0),
            (1.5, 200.0, 300.0),
            (3.0, 50.0, 150.0),
        ]
        
        for revenue_rate, order_amount, expected_revenue in test_cases:
            company = Company(
                capital=10000.0,
                sector_id=0,
                location=(50.0, 50.0),
                revenue_rate=revenue_rate
            )
            company.add_revenue(order_amount)
            
            assert company.revenue == expected_revenue, \
                f"Formula failed: {revenue_rate} × {order_amount} = {company.revenue}, expected {expected_revenue}"
        
        print("✅ Revenue calculation formula verified: revenue = revenue_rate × order_amount")


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("REVENUE RATE TEST SUITE")
    print("="*70)
    
    test_suite = TestRevenueRate()
    
    try:
        test_suite.test_default_revenue_rate()
        test_suite.test_doubled_revenue_rate()
        test_suite.test_halved_revenue_rate()
        test_suite.test_multiple_orders_accumulate()
        test_suite.test_revenue_rate_in_environment()
        test_suite.test_revenue_calculation_formula()
        
        print("\n" + "="*70)
        print("✅ ALL REVENUE RATE TESTS PASSED")
        print("="*70)
        print("\nRevenue rate mechanism verified:")
        print("  ✓ Formula: revenue = revenue_rate × order_amount")
        print("  ✓ Multiple orders accumulate correctly")
        print("  ✓ Environment properly configures companies")
        print("  ✓ Allows tuning revenue efficiency")
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
