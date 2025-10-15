#!/usr/bin/env python3
"""
Master test runner for Industry Simulation test suite.

Runs all test modules and provides a comprehensive summary.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all test modules
from tests.test_config_integration import TestConfigIntegration
from tests.test_death_threshold import TestDeathThreshold
from tests.test_logistic_cost import TestLogisticCost
from tests.test_hyperparameters import TestHyperparameters
from tests.test_revenue_rate import TestRevenueRate
from tests.test_product_system import TestProductSystem
from tests.test_multi_action import TestMultiActionSystem


def run_test_suite(suite_name, test_class):
    """Run a test suite and return success status."""
    print("\n" + "="*70)
    print(f"Running {suite_name}")
    print("="*70)
    
    try:
        suite = test_class()
        # Get all test methods
        test_methods = [method for method in dir(suite) if method.startswith('test_')]
        
        passed = 0
        failed = 0
        
        for method_name in test_methods:
            try:
                method = getattr(suite, method_name)
                method()
                passed += 1
            except AssertionError as e:
                print(f"\n❌ {method_name} FAILED: {e}")
                failed += 1
            except Exception as e:
                print(f"\n❌ {method_name} ERROR: {e}")
                failed += 1
        
        print(f"\n{suite_name}: {passed} passed, {failed} failed")
        return failed == 0
    
    except Exception as e:
        print(f"\n❌ {suite_name} could not run: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all test suites."""
    print("\n" + "="*70)
    print("INDUSTRY SIMULATION - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print("\nRunning all test modules...")
    
    results = {}
    
    # Run all test suites
    results['Config Integration'] = run_test_suite(
        'Config Integration Tests',
        TestConfigIntegration
    )
    
    results['Death Threshold'] = run_test_suite(
        'Death Threshold Tests',
        TestDeathThreshold
    )
    
    results['Logistic Cost'] = run_test_suite(
        'Logistic Cost Tests',
        TestLogisticCost
    )
    
    results['Hyperparameters'] = run_test_suite(
        'Hyperparameter Tests',
        TestHyperparameters
    )
    
    results['Revenue Rate'] = run_test_suite(
        'Revenue Rate Tests',
        TestRevenueRate
    )
    
    results['Product System'] = run_test_suite(
        'Product System Tests',
        TestProductSystem
    )
    
    results['Multi-Action System'] = run_test_suite(
        'Multi-Action System Tests',
        TestMultiActionSystem
    )
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total_passed = sum(1 for v in results.values() if v)
    total_failed = sum(1 for v in results.values() if not v)
    
    for suite_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{suite_name:30} {status}")
    
    print("\n" + "="*70)
    print(f"Total: {total_passed}/{len(results)} test suites passed")
    print("="*70)
    
    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nThe Industry Simulation codebase is working correctly:")
        print("  ✓ Configuration system integrated")
        print("  ✓ Death threshold mechanism working")
        print("  ✓ Logistic costs calculated correctly")
        print("  ✓ All hyperparameters respected")
        print("  ✓ Revenue rate system functional")
        print("  ✓ Product system operational")
        print("  ✓ Multi-action system working correctly")
        print()
        return 0
    else:
        print(f"\n⚠️  {total_failed} test suite(s) failed")
        print("Please review the errors above and fix the issues.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
