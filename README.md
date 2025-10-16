# Industry Simulation Environment

A reinforcement learning environment for simulating spatial economic dynamics with supply chains, logistic costs, and multi-action control.

**Key Features:**

- **Spatial economics**: Companies positioned on 2D map with real company data
- **Supply chain simulation**: 7-sector automotive supply chain (Raw → Parts → Electronics → Battery/Motor → OEM → Service)
- **Real company data**: 927 companies with establishment dates, active status, and death dates
- **Logistic costs**: Distance-based costs using **inverse square law** (cost ∝ 1/d²)
- **Multi-action system**: Agents can perform 0-10 actions per step
- **Flexible configuration**: JSON-based config system with 100+ parameters in organized config/ package
- **Product-based trading**: Realistic supply chain with product generation and consumption
- **Revenue rate system**: Automatic revenue for OEM and Service sectors
- **Death threshold**: Companies with low capital go bankrupt
- **Environment persistence**: Save/load complete simulation states
- **Visualization**: Interactive plots with mplcursors, automatic periodic dashboards at 600 DPI
- **Training infrastructure**: Supports custom and Stable-Baselines3 training

### 🤖 RL-Ready Environment

- Gymnasium-compatible interface
- Multi-action Dict space (invest vs create, 0-10 actions per step)
- Aggregate observations for scalability
- Support for Stable-Baselines3 and Ray RLlib
- Detailed action results tracking

### 📊 Rich Visualization & Environment Persistence

- Interactive matplotlib plots with mplcursors hover information
- 6 different visualization types
- Comprehensive dashboard view
- **Automatic periodic visualization** during simulation
- **Environment saving and loading** for analysis and checkpointing
- **PNG export** of all visualizations
- Real-time training monitoring

### 🧪 Complete Test Suite

- 33 tests covering all functionality
- Test helper functions for easy action creation
- Comprehensive test documentation
- Tests for multi-action, configuration, hyperparameters, and more

### ⚙️ Configuration System

- JSON-based hyperparameter management
- Easy experiment reproduction
- Pre-configured setups for different scenarios
- Support for ablation studies

## 🚀 Quick Start

### Installation

```bash
# Option 1: Use uv sync (recommended - installs all dependencies from pyproject.toml)
uv sync

# Option 2: Install dependencies individually
uv pip install gymnasium numpy matplotlib tqdm rich

# Optional: Install RL framework
uv pip install stable-baselines3 tensorboard
# OR
uv pip install ray[rllib]
```

### Train RL Agent

```bash
# Train with Stable-Baselines3 (PPO)
uv run python main.py --framework sb3 --algorithm ppo --timesteps 100000

# View help for all options
uv run python main.py --help
```

### Run Tests

```bash
# Run all tests
uv run tests/run_all_tests.py

# Or use pytest
pytest tests/

# Run specific test file
uv run tests/test_multi_action.py
```

### Process Company Data

```bash
# Generate company classification data from Excel source
cd data
uv run python supply_chain.py

# Outputs: company_classification.json and company_classification.csv
# 7 fields per company: name, sector, size, capital, establishment_date, active, death_date
```

## 📊 Real Company Data

This simulation uses **real company data** from 927 companies in the automotive supply chain:

### Company Fields (7 total)

1. **company_name** - Company legal name
2. **sector** - Industry sector (Raw, Parts, Electronics, Battery/Motor, OEM, Service, Other)
3. **company_size** - Size classification (large, medium, small, micro)
4. **initial_capital** - Registered capital amount
5. **establishment_date** - Company founding date (成立日期)
6. **active** - Boolean status (true = active, false = inactive)
7. **death_date** - Optional closure date (注销日期, if company became inactive)

### Statistics

- **Total companies**: 927
- **Active companies**: 361 (38.9%)
- **Inactive companies**: 566 (61.1%)
- **Companies with death dates**: 339 (36.6% of total)

### Data Source and Processing

- **Origin**: Excel data from Chinese company registration records
- **Processing**: `data/supply_chain.py` with regex-based date extraction
- **Pattern**: Extracts dates from status strings like "注销（2017-06-16）"
- **Output formats**: JSON and CSV for simulation integration

**For detailed data pipeline documentation**, see [`data/README.md`](data/README.md).

## 💡 Multi-Action System

The agent can now perform **multiple actions in a single step**!

### Action Format

```python
action = {
    "num_actions": 3,  # Choose 0 to max_actions_per_step
    "actions": [
        {"op": 0, "invest": {"firm_id": 0, "amount": [1000.0]}, ...},  # Invest
        {"op": 0, "invest": {"firm_id": 1, "amount": [1500.0]}, ...},  # Invest
        {"op": 1, "create": {"sector": 2, "initial_capital": [30000.0], ...}, ...},  # Create
    ]
}
```

### Benefits

- **Efficiency**: Make multiple decisions per step
- **Strategy**: Balance between invest and create actions
- **Flexibility**: Choose 0-10 actions (configurable)
- **Performance**: Much faster than single-action per step

See `MULTI_ACTION_SYSTEM.md` and `demo_reward_system.py` for details.

## 📖 Command-Line Usage

### Basic Training

```bash
# Default training (100K steps)
uv run python main.py --framework sb3 --algorithm ppo --timesteps 100000

# Advanced training with custom parameters
uv run python main.py --framework sb3 --algorithm ppo \
  --timesteps 500000 \
  --num-envs 8 \
  --lr 0.0003 \
  --initial-firms 15 \
  --save-freq 20000

# Evaluate trained model
uv run python main.py --mode eval \
  --model-path ./checkpoints/sb3_ppo_*/ppo_final.zip \
  --framework sb3

# Demo with visualization
uv run python main.py --mode demo \
  --model-path ./checkpoints/sb3_ppo_*/ppo_final.zip \
  --framework sb3
```

### Using Configuration Files

```bash
# Use default config
uv run python main.py --config config.json

# Use custom config
uv run python main.py --config my_experiment.json

# Override specific parameters
uv run python main.py --config config.json \
  --logistic-cost-rate 200.0 \
  --death-threshold 10000
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--framework` | `sb3` | RL framework: `sb3`, `rllib`, or `custom` |
| `--algorithm` | `ppo` | Algorithm: `ppo`, `a2c`, or `sac` |
| `--timesteps` | `100000` | Total training timesteps |
| `--num-envs` | `4` | Number of parallel environments |
| `--lr` | `3e-4` | Learning rate |
| `--initial-firms` | `10` | Starting number of companies |
| `--mode` | `train` | Mode: `train`, `eval`, or `demo` |

Run `uv run python main.py --help` for full list.

## 🔧 Key Hyperparameters

### max_actions_per_step (NEW!)

**Default**: 10  
**Range**: 1-20  
Controls how many actions the agent can take per step.

### logistic_cost_rate

**Default**: 100.0  
**Range**: 0-1000+  
Controls distance-based transportation costs using inverse square law: `cost = rate × volume / distance²`

**Effect**:

- `0`: No logistic costs (ablation baseline)
- `100`: Moderate clustering incentive (default)
- `500`: Strong clustering pressure
- `1000+`: Forced clustering

### revenue_rate

**Default**: 1.0  
**Range**: 0.5-2.0  
Multiplier for converting orders to revenue: `revenue = revenue_rate × order_amount`

**Interpretation**:

- `< 1.0`: Low profit margins (retail, commodities)
- `= 1.0`: Standard margins (default)
- `> 1.0`: High profit margins (tech, finance)

### death_threshold

**Default**: 0.0  
**Range**: 0-50000+  
Companies with `capital < death_threshold` are removed (bankruptcy).

**Settings**:

- `0.0`: No deaths, companies can have negative capital
- `5000-10000`: Realistic business environment
- `20000+`: Only profitable companies survive

## 💾 Environment Persistence & Visualization

### Automatic Periodic Visualization

Configure the environment to automatically save visualizations during simulation:

```python
from config import load_config
from env import IndustryEnv

config = load_config("config/config.json")
config.environment.visualize_every_n_steps = 100  # Visualize every 100 steps
config.environment.visualization_dir = "visualizations/training"

env = IndustryEnv(config.environment)
obs, info = env.reset(options={"initial_firms": 20})

# Visualizations are automatically saved as PNGs at steps 100, 200, 300, ...
for step in range(1000):
    action = get_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
```

### Save Environment State

```python
# Save with automatic timestamp
filepath = env.save_environment()  # Saves as environment_YYYYMMDD_HHMMSS.pkl

# Save with custom filename
filepath = env.save_environment("checkpoints/my_simulation.pkl")
```

### Load and Visualize Saved Environments

```python
from env import load_environment, visualize_saved_environment

# Load a saved environment
env = load_environment("environment_20240101_120000.pkl")
print(f"Loaded {env.num_firms} firms at step {env.current_step}")

# Create interactive visualization
visualize_saved_environment(
    "environment_20240101_120000.pkl",
    output_dir="analysis"
)
# Opens interactive matplotlib window with mplcursors hover info
```

### Manual Visualization

```python
# Visualize at specific checkpoints
env.visualize_step("before_major_event")
# ... simulation ...
env.visualize_step("after_major_event")
```

**Features:**

- **Interactive hover**: mplcursors provides company details on hover
- **PNG export**: All visualizations saved as high-DPI PNG files
- **Comprehensive dashboard**: Company locations, sector distribution, capital heatmap
- **Configurable**: Control frequency, directory, format, and display options

See `example_save_visualize.py` and `VISUALIZATION_FEATURES.md` for detailed examples.

## 🧪 Examples

### Basic Environment Usage

```python
from env import IndustryEnv
from tests import create_single_action

# Create environment
env = IndustryEnv()
obs, info = env.reset(options={"initial_firms": 10})

# Single action (using helper)
action = create_single_action(
    op=1,  # Create new company
    invest_dict={"firm_id": 0, "amount": [0.0]},
    create_dict={
        "initial_capital": [50000.0],
        "sector": 2,
        "location": [50.0, 50.0]
    },
    max_actions=env.max_actions_per_step
)

obs, reward, terminated, truncated, info = env.step(action)
print(f"Reward: {reward}, Firms: {info['num_firms']}")
print(f"Actions taken: {info['num_actions']}")
print(f"Valid actions: {info['num_valid_actions']}")
```

### Multiple Actions

```python
# Multiple actions in one step
action = {
    "num_actions": 3,
    "actions": [
        {"op": 0, "invest": {"firm_id": 0, "amount": [1000.0]}, ...},
        {"op": 0, "invest": {"firm_id": 1, "amount": [2000.0]}, ...},
        {"op": 1, "create": {"sector": 5, "initial_capital": [40000.0], ...}, ...}
    ] + [dummy_action] * 7  # Fill to max_actions_per_step
}

obs, reward, terminated, truncated, info = env.step(action)
print(f"Total reward: {reward}")
print(f"Action results: {info['action_results']}")
```

### Using Configuration

```python
from config import Config
from env import IndustryEnv

# Load config
config = Config.from_json("config.json")

# Create environment with config
env = IndustryEnv(
    size=config.environment.size,
    max_actions_per_step=config.environment.max_actions_per_step,
    logistic_cost_rate=config.environment.logistic_cost_rate
)

obs, info = env.reset(options={
    "initial_firms": config.environment.initial_firms
})
```

## 📊 Monitoring Training

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir=./logs

# View at http://localhost:6006
```

### Output Structure

Training creates organized outputs:

```plaintext
./logs/sb3_ppo_20251014_120000/
├── config.json                 # Training configuration
├── events.out.tfevents.*       # TensorBoard logs
├── training_history.json       # Episode history
└── dashboard_episode_10.png    # Visualizations

./checkpoints/sb3_ppo_20251014_120000/
├── ppo_model_10000_steps.zip   # Periodic checkpoints
├── ppo_final.zip               # Final model
└── best_model/
    └── best_model.zip          # Best performing model
```

## 📂 Project Structure

```plaintext
industry-sim/
├── env.py                      # Environment definition
├── config.py                   # Configuration system
├── main.py                     # Training scaffolding
├── visualize.py                # Visualization tools
├── spatial_utils.py            # Spatial analysis
├── config.json                 # Main configuration
│
├── tests/                      # Test suite
│   ├── __init__.py            # Test helpers
│   ├── run_all_tests.py       # Test runner
│   ├── test_multi_action.py   # Multi-action tests
│   ├── test_config_integration.py
│   ├── test_hyperparameters.py
│   ├── test_death_threshold.py
│   ├── test_logistic_cost.py
│   └── test_revenue_rate.py
│
├── example_usage.py            # Usage examples
├── example_visualize.py        # Visualization examples
├── demo_reward_system.py       # Reward system demo
│
├── README.md                   # This file
└── DETAILS.md                  # Comprehensive docs
```

## 🧪 Testing

### Test Suite

The project includes **33 comprehensive tests**:

- **Multi-Action Tests** (5): Zero, single, multiple, max, mixed actions
- **Config Integration** (4): Config loading and environment creation
- **Hyperparameters** (9): All hyperparameters respected
- **Death Threshold** (4): Bankruptcy mechanism
- **Logistic Cost** (5): Distance-based costs
- **Revenue Rate** (6): Revenue calculation

### Running Tests

```bash
# Run all tests
uv run tests/run_all_tests.py

# Run specific test suite
uv run tests/test_multi_action.py
uv run tests/test_hyperparameters.py

# Use pytest
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

All tests pass! ✅

## 📚 Documentation

- **`README.md`** (this file) - Quick start guide and overview
- **`DETAILS.md`** - Comprehensive technical documentation with all implementation details
- **`data/README.md`** - Data pipeline, company fields, and processing scripts
- **`demos/README.md`** - Demo scripts and usage examples
- **`tests/README.md`** - Testing framework and test suite
- **`config/config.json`** - Full hyperparameter reference with all configuration options

### In DETAILS.md

- Complete environment specification
- Logistic cost system details (inverse square law)
- Revenue rate system
- Death threshold mechanism
- Configuration system guide
- Training guide and best practices
- Visualization system
- Spatial analysis tools
- API reference
- Troubleshooting

## 🎓 Key Results

### Multi-Action Performance

- **Efficiency**: 3-5x faster than single-action equivalent
- **Flexibility**: Agent chooses optimal action count per situation
- **Strategy**: Can balance investments and creations

### Logistic Cost Impact

Companies benefit from **clustering** near supply chain partners:

- **Cost reduction**: ~75% savings with optimal clustering
- **Strategic depth**: Location becomes critical decision
- **Emergent behavior**: Industrial districts form naturally

### Training Performance

| Configuration | Time (100K steps) | Notes |
|---------------|-------------------|-------|
| 1 env, CPU | ~10-15 min | Baseline |
| 4 envs, CPU | ~5-8 min | Default |
| 8 envs, CPU | ~3-5 min | Recommended |
| 16 envs, GPU | ~2-3 min | Fastest |

## 🆘 Troubleshooting

**Import errors?**

```bash
uv pip install gymnasium numpy matplotlib
uv pip install stable-baselines3 tensorboard
```

**Tests failing?**

- Ensure you're using the test helper: `from tests import create_single_action`
- Check that you're passing `max_actions` parameter
- See `tests/README.md` for details

**Action format errors?**

- Use the new multi-action format (see examples above)
- Use `create_single_action()` helper for single actions
- See `MULTI_ACTION_SYSTEM.md` for migration guide

**Low performance?**

- Increase `--num-envs` (e.g., 8 or 16)
- Decrease `--initial-firms` for faster steps
- Use GPU if available

**Want to disable features?**

```bash
# Disable logistic costs
uv run python main.py --logistic-cost-rate 0.0

# Disable death mechanism
uv run python main.py --death-threshold 0.0
```

**Ready to train intelligent agents in a spatial economic simulation with multi-action capabilities!** 🎯

For detailed documentation, examples, and API reference, see **`DETAILS.md`**.
