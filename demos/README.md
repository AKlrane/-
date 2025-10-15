# Demonstration Scripts

This directory contains interactive demonstration scripts that showcase various features of the Industry Simulation environment.

## Available Demos

### Core Feature Demos

#### 1. demo_reward_system.py
Demonstrates the reward calculation system with multi-action support.

**Features**:
- Shows reward breakdown for different action types
- Demonstrates single and multiple actions
- Displays action results tracking
- Compares valid vs invalid actions

**Run**:
```bash
python demos/demo_reward_system.py
```

#### 2. demo_logistic_cost.py
Visual demonstration of distance-based logistic costs.

**Features**:
- Plots cost vs distance curves
- Shows clustering effects on costs
- Compares dispersed vs clustered layouts
- Demonstrates inverse square law

**Run**:
```bash
python demos/demo_logistic_cost.py
```

#### 3. demo_revenue_rate.py
Shows how revenue_rate affects company profitability.

**Features**:
- Compares different revenue rates (0.5, 1.0, 2.0)
- Shows revenue calculation formula
- Demonstrates profit margin effects
- Configuration examples

**Run**:
```bash
python demos/demo_revenue_rate.py
```

#### 4. demo_death_threshold.py
Demonstrates the company death/bankruptcy mechanism.

**Features**:
- Shows companies dying when capital falls below threshold
- Tracks death statistics over time
- Demonstrates survival analysis
- Shows configuration effects

**Run**:
```bash
python demos/demo_death_threshold.py
```

#### 5. demo_product_system.py
Demonstrates the product system and supply chain dynamics.

**Features**:
- Shows tier-based production (Tier 0 produces from scratch)
- Demonstrates downstream purchasing from upstream
- Product inventory tracking across companies
- Supply chain network formation
- Revenue generation through product sales

**Run**:
```bash
python demos/demo_product_system.py
```

#### 6. demo_tier_calculation.py
Shows dynamic tier calculation from sector relationships.

**Features**:
- Displays sector relationships (suppliers/consumers)
- Shows automatic tier assignment using topological sorting
- Visualizes supply chain flow across tiers
- Demonstrates automotive supply chain model (7 sectors)
- Explains tier types (root, intermediate, leaf, isolated)

**Run**:
```bash
python demos/demo_tier_calculation.py
```

### Example Usage Demos

#### 7. demo_usage.py
Basic usage example of the IndustryEnv environment.

**Features**:
- Shows how to create and reset environment
- Demonstrates action space usage
- Random action sampling
- Location-based features
- Company information display

**Run**:
```bash
python demos/demo_usage.py
```

#### 8. demo_visualize.py
Demonstrates visualization capabilities.

**Features**:
- Interactive company location maps with hover details
- Sector distribution charts
- Capital distribution heatmaps
- Comprehensive dashboard view
- Spatial insights and analysis

**Run**:
```bash
python demos/demo_visualize.py
```

#### 9. demo_config.py
Configuration system usage examples.

**Features**:
- Load and use config/config.json
- Create environments from config
- Run episodes with config parameters
- Ablation studies (enable/disable features)
- Hyperparameter sweeps
- Save custom configurations to config/ folder

**Run**:
```bash
python demos/demo_config.py
```

## Running Demos

All demos can be run from the project root directory:

```bash
# Core feature demos
python demos/demo_reward_system.py
python demos/demo_logistic_cost.py
python demos/demo_revenue_rate.py
python demos/demo_death_threshold.py
python demos/demo_product_system.py
python demos/demo_tier_calculation.py

# Example usage demos
python demos/demo_usage.py
python demos/demo_visualize.py
python demos/demo_config.py
```

## Demo Purpose

These demos are educational tools to help understand:
- How different hyperparameters affect the simulation
- The multi-action system
- Reward calculation
- Economic dynamics (costs, revenues, deaths)
- Product system and supply chain mechanics
- Dynamic tier calculation
- Visualization capabilities
- Configuration system usage

## Requirements

All demos require the standard project dependencies:
- gymnasium
- numpy
- matplotlib (for visual demos: demo_visualize.py, demo_logistic_cost.py)

Install with:
```bash
uv pip install gymnasium numpy matplotlib
```

Or using the project's pyproject.toml:
```bash
uv sync
```

## Notes

- All demos automatically add parent directory to path for imports
- Demos work with the latest configuration system (config/config.json)
- Visual demos display interactive plots using matplotlib
- Each demo includes explanatory console output
- Demos reflect the current 7-sector automotive supply chain model
- Product system demos show Tier 0-4 supply chain dynamics

## Demo Categories

**Core Features**: Reward system, logistics, revenue, death threshold  
**Supply Chain**: Product system, tier calculation  
**Usage Examples**: Basic usage, visualization, configuration  

## Latest Updates

- ✅ All demos moved to `/demos` folder
- ✅ Updated to use Config objects properly
- ✅ Compatible with 7-sector automotive model
- ✅ Updated to use action_space.sample()
- ✅ Fixed imports for subdirectory structure
