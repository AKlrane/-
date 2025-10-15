# Visualization Examples Guide

This guide explains the comprehensive visualization example that combines all visualization capabilities of the Industry Simulation.

## Running the Examples

```bash
python demos/example_visualize.py
```

## Available Examples

### 1. Basic Interactive Visualization
**Quick demo with 4 plot types**
- Interactive company location map with hover details
- Sector distribution bar chart
- Capital distribution heatmap
- Comprehensive dashboard with statistics

**Use case:** Quick overview and exploration of simulation state

### 2. Periodic Visualization
**Auto-save plots during long simulation**
- Runs a 500-step simulation
- Automatically saves visualizations every 100 steps
- No blocking - runs continuously
- Saves to `visualizations/periodic/`

**Use case:** Long training runs where you want to monitor progress without interruption

### 3. Save & Load Environment
**Persist and restore simulation state**
- Saves environment state to pickle files
- Loads saved environments for analysis
- Creates visualizations from saved states
- Saves to `saved_envs/`

**Use case:** Checkpointing experiments, sharing results, post-simulation analysis

### 4. Manual Visualization
**Control exactly when to create plots**
- Runs simulation without automatic visualization
- Manually trigger visualization at specific checkpoints
- Fine-grained control over when plots are created
- Saves to `visualizations/manual/`

**Use case:** Creating plots only at key moments (e.g., before/after major events)

### 5. Run All Examples
**Execute all demonstrations in sequence**
- Runs examples 1, 2, 3, and 4 automatically
- Skips the load & visualize step to avoid blocking
- Shows comprehensive summary at the end

**Use case:** Full demonstration of all capabilities

## Interactive Features

All visualizations include:
- **Hover tooltips** - Mouse over companies to see details
- **Color coding** - Each sector has a unique color
- **Size proportional to capital** - Larger circles = more capital
- **Grid overlays** - Helps with spatial analysis

## Output Directories

Generated files are organized as follows:

```
visualizations/
├── periodic/          # Auto-generated during simulation
├── manual/            # Manually triggered visualizations
└── loaded/            # Visualizations from loaded environments

saved_envs/            # Saved environment pickle files
```

## Programmatic Usage

### Basic Visualization
```python
from utils import create_dashboard, plot_companies
import matplotlib.pyplot as plt

# Create dashboard
fig = create_dashboard(env)
plt.show()

# Or individual plots
plot_companies(env.companies, size=env.size, show_labels=True)
plt.show()
```

### Saving and Loading
```python
from env import IndustryEnv, load_environment, visualize_saved_environment

# Save
filepath = env.save_environment("my_simulation.pkl")

# Load
loaded_env = load_environment("my_simulation.pkl")

# Visualize loaded environment
visualize_saved_environment("my_simulation.pkl", output_dir="my_viz")
```

### Periodic Visualization
```python
from config import load_config
from env import IndustryEnv

config = load_config("config/config.json")
config.environment.visualize_every_n_steps = 100
config.environment.visualization_dir = "my_visualizations"
config.environment.save_plots = True
config.environment.show_plots = False

env = IndustryEnv(config.environment)
# Visualizations will be automatically created every 100 steps
```

## Tips

1. **For training runs**: Use Example 2 (Periodic Visualization) with `show_plots=False`
2. **For analysis**: Use Example 1 (Basic Visualization) with interactive plots
3. **For experiments**: Use Example 3 (Save & Load) to preserve states
4. **For presentations**: Use Example 4 (Manual Visualization) to control timing

## Recent Updates

- **Statistics Panel Enhanced**: Now includes max/min capital metrics and better formatting
- **Combined Examples**: Merged `example_save_visualize.py` and `example_visualize.py` into one comprehensive script
- **Menu Interface**: Added interactive menu for easy example selection
