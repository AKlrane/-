#!/bin/bash
# Quick test runner script for Industry Simulation

echo "🧪 Running Industry Simulation Test Suite"
echo "=========================================="
echo ""

# Parse command line arguments
RUN_SPECIFIC=""
VERBOSE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -t|--test)
            RUN_SPECIFIC="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -v, --verbose          Show verbose output"
            echo "  -t, --test <name>      Run specific test (e.g., config, logistic, multi_action)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Available tests:"
            echo "  config              - Configuration integration tests"
            echo "  death_threshold     - Death threshold system tests"
            echo "  hyperparameters     - Hyperparameter validation tests"
            echo "  logistic            - Logistic cost system tests (inverse square law)"
            echo "  multi_action        - Multi-action system tests (0-10 actions per step)"
            echo "  product             - Product system and supply chain tests"
            echo "  revenue             - Revenue rate system tests"
            echo "  sector_costs        - Sector-specific operating cost tests"
            echo ""
            echo "Examples:"
            echo "  ./run_tests.sh                    # Run all tests"
            echo "  ./run_tests.sh -v                 # Run all tests with verbose output"
            echo "  ./run_tests.sh -t logistic        # Run only logistic cost tests"
            echo "  ./run_tests.sh -t multi_action -v # Run multi-action tests verbosely"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Run specific test if requested
if [ ! -z "$RUN_SPECIFIC" ]; then
    echo "Running specific test: $RUN_SPECIFIC"
    echo ""
    
    case $RUN_SPECIFIC in
        config)
            uv run tests/test_config_integration.py
            ;;
        death_threshold|death)
            uv run tests/test_death_threshold.py
            ;;
        hyperparameters|hyper)
            uv run tests/test_hyperparameters.py
            ;;
        logistic)
            uv run tests/test_logistic_cost.py
            ;;
        multi_action|multi)
            uv run tests/test_multi_action.py
            ;;
        product|products)
            uv run tests/test_product_system.py
            ;;
        revenue)
            uv run tests/test_revenue_rate.py
            ;;
        sector_costs|sector)
            uv run tests/test_sector_operating_costs.py
            ;;
        *)
            echo "❌ Unknown test: $RUN_SPECIFIC"
            echo "Use -h for available tests"
            exit 1
            ;;
    esac
    
    TEST_EXIT_CODE=$?
else
    # Run all tests
    uv run tests/run_all_tests.py
    TEST_EXIT_CODE=$?
fi

# Check exit code
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ All tests passed successfully!"
    echo ""
    echo "📋 Available test suites:"
    echo "  • test_config_integration       - Config system and package structure"
    echo "  • test_death_threshold          - Company bankruptcy mechanics"
    echo "  • test_hyperparameters          - Parameter validation and ranges"
    echo "  • test_logistic_cost            - Inverse square law logistics (cost ∝ 1/d²)"
    echo "  • test_multi_action             - Multi-action system (0-10 actions/step)"
    echo "  • test_product_system           - Product generation and supply chain"
    echo "  • test_revenue_rate             - Automatic revenue for OEM/Service sectors"
    echo "  • test_sector_operating_costs   - Sector-specific cost multipliers (0.8x-1.5x)"
    echo ""
    echo "🔍 Run individual tests with:"
    echo "  ./run_tests.sh -t <test_name>    (e.g., ./run_tests.sh -t logistic)"
    echo "  uv run tests/test_<name>.py      (e.g., uv run tests/test_logistic_cost.py)"
    echo ""
    echo "📚 For more information:"
    echo "  • See tests/README.md for detailed test documentation"
    echo "  • Use ./run_tests.sh -h for command-line options"
else
    echo ""
    echo "❌ Some tests failed. Check output above."
    echo ""
    echo "💡 Debugging tips:"
    echo "  • Run specific test: ./run_tests.sh -t <test_name>"
    echo "  • Check test output for detailed error messages"
    echo "  • Verify config files are up-to-date"
    echo "  • Ensure all dependencies are installed: uv sync"
    exit 1
fi
