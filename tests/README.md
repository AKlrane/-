# Industry Simulation Test Suite

This directory contains all tests for the Industry Simulation environment.

## Test Structure

All tests have been consolidated into this single `tests/` directory for better organization and maintainability.

### Test Modules

1. **`test_config_integration.py`** - Tests configuration system integration
   - Config loading from JSON
   - Environment creation with config
   - Custom config files
   - Hyperparameter propagation

2. **`test_death_threshold.py`** - Tests company death/bankruptcy mechanism
   - No death threshold (negative capital allowed)
   - Positive death threshold
   - High death threshold (aggressive culling)
   - `allow_negative_capital` flag

3. **`test_logistic_cost.py`** - Tests logistic cost calculations
   - Inverse square law: `cost = k * volume / distance²`
   - Different logistic cost rates
   - Disable logistic costs flag
   - Distance impact on profitability
   - Min distance epsilon (prevents division by zero)

4. **`test_hyperparameters.py`** - Tests that all hyperparameters are respected
   - Spatial parameters (size, sectors, episode steps)
   - Capacity parameters (max companies, initial firms)
   - Company parameters (costs, capital ranges, death)
   - Supply chain parameters (trade, revenue, epsilon)
   - Logistics parameters (costs, disable flag)
   - Reward parameters (multipliers, penalties)
   - Ablation flags (disable flags, fixed locations)
   - Investment constraints (min/max)

5. **`test_revenue_rate.py`** - Tests revenue calculation system
   - Formula: `revenue = revenue_rate × order_amount`
   - Multiple orders accumulation
   - Environment configuration

6. **`test_product_system.py`** - Tests product-based supply chain system
   - Product parameter initialization
   - Supply chain tier assignment (0-3, upstream to downstream)
   - Supplier-customer network formation
   - Tier 0 production (produce from scratch)
   - Downstream purchasing (buy from suppliers)
   - Production capacity constraints (`capital × ratio`)
   - Purchase budget constraints (`capital × ratio`)
   - Product inventory tracking
   - Info metrics (inventory, produced, sold, purchased)
   - Backward compatibility (products disabled)

## Running Tests

### Run All Tests

```bash
# From the industry-sim directory
python tests/run_all_tests.py
```

Or from the tests directory:

```bash
cd tests
python run_all_tests.py
```

### Run Individual Test Modules

```bash
# From the industry-sim directory
python tests/test_config_integration.py
python tests/test_death_threshold.py
python tests/test_logistic_cost.py
python tests/test_hyperparameters.py
python tests/test_revenue_rate.py
python tests/test_product_system.py
```

### Using pytest (if installed)

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_config_integration.py

# Run with verbose output
pytest tests/ -v

# Run specific test class
pytest tests/test_config_integration.py::TestConfigIntegration

# Run specific test method
pytest tests/test_config_integration.py::TestConfigIntegration::test_config_loading
```

## Test Coverage

The test suite covers:

- ✅ Configuration loading and validation
- ✅ Environment initialization with configs
- ✅ Company creation and parameter propagation
- ✅ Death/bankruptcy mechanisms
- ✅ Logistic cost calculations (inverse square law)
- ✅ Supply chain interactions
- ✅ Revenue calculations
- ✅ Reward calculations
- ✅ Episode management
- ✅ Ablation study flags
- ✅ All hyperparameters from `config.json`
- ✅ Product system (production, purchasing, inventory)
- ✅ Supply chain network formation (tiers, suppliers, customers)

## Requirements

The tests require:
- Python 3.8+
- numpy
- gymnasium
- The main codebase modules: `config.py`, `env.py`

Optional:
- pytest (for advanced test running features)

## Test Philosophy

1. **Comprehensive**: Tests cover all major features and hyperparameters
2. **Updated**: Tests reflect the newest version of the codebase
3. **Organized**: All tests in one directory with clear naming
4. **Independent**: Tests can run individually or as a suite
5. **Informative**: Clear output showing what's being tested
6. **Fast**: Tests run quickly for rapid iteration

## Adding New Tests

When adding new features to the simulation:

1. Create a new test file: `tests/test_<feature_name>.py`
2. Follow the existing test structure:

   ```python
   class Test<FeatureName>:
       def test_<specific_aspect>(self):
           # Test implementation
           assert condition, "Error message"
   ```

3. Add the test to `run_all_tests.py`
4. Update this README

All tests have been updated to work with the current version of the codebase.
