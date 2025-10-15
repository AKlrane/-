"""
Comprehensive example demonstrating visualization and environment management features.

This script combines multiple visualization capabilities:
1. Interactive visualization with various plot types
2. Periodic visualization during simulation
3. Environment saving and loading
4. Manual visualization control at specific checkpoints
5. Spatial analysis and insights
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from config import Config, load_config
from env import IndustryEnv, load_environment, visualize_saved_environment, sector_relations
from utils import (
    plot_companies,
    plot_sector_distribution,
    plot_capital_distribution,
    create_dashboard
)
import matplotlib.pyplot as plt


def example_basic_visualization():
    """Example 1: Basic interactive visualization."""
    print("=" * 60)
    print("Example 1: Basic Interactive Visualization")
    print("=" * 60)
    
    # Create environment and run some steps
    config = Config()
    env = IndustryEnv(config.environment)
    obs, _ = env.reset(seed=42, options={"initial_firms": 10})
    
    print(f"\nStarting with {obs['num_firms']} companies")
    print(f"Total capital: ${obs['total_capital'][0]:,.2f}\n")
    
    # Run simulation for several steps with random actions
    print("Running 20 simulation steps...")
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\nAfter 20 steps:")
    print(f"  Companies: {obs['num_firms']}")
    print(f"  Total capital: ${obs['total_capital'][0]:,.2f}")
    print(f"  Sector distribution: {obs['sector_counts']}")
    
    # Create visualizations
    print("\n=== Creating Visualizations ===\n")
    
    # 1. Main company location plot with interactive hover
    print("1. Interactive company location map...")
    fig1 = plot_companies(
        env.companies,
        size=env.size,
        show_labels=True,
        title=f"Company Locations (Step {env.current_step})"
    )
    
    # 2. Sector distribution bar chart
    print("2. Sector distribution chart...")
    fig2 = plot_sector_distribution(env.companies)
    
    # 3. Capital heatmap
    print("3. Capital distribution heatmap...")
    fig3 = plot_capital_distribution(env.companies, size=env.size, grid_size=15)
    
    # 4. Comprehensive dashboard
    print("4. Complete dashboard...")
    fig4 = create_dashboard(env)
    
    print("\n" + "="*50)
    print("Visualizations ready!")
    print("="*50)
    print("\nInteractive features:")
    print("  - Hover over points in the location map to see company details")
    print("  - Company size indicates capital amount")
    print("  - Colors represent different sectors")
    print("\nClose the plot windows to continue.")
    
    # Show all plots
    plt.show()
    
    # Print some insights
    print("\n=== Spatial Insights ===")
    
    # Calculate average distance between companies
    total_distance = 0
    count = 0
    for i, c1 in enumerate(env.companies):
        for c2 in env.companies[i+1:]:
            total_distance += c1.distance_to(c2)
            count += 1
    
    if count > 0:
        avg_distance = total_distance / count
        print(f"Average distance between companies: {avg_distance:.2f} units")
    
    # Find most capital-rich sector
    sector_capital = {}
    for company in env.companies:
        sector_capital[company.sector_id] = sector_capital.get(company.sector_id, 0) + company.capital
    
    if sector_capital:
        richest_sector = max(sector_capital.items(), key=lambda x: x[1])
        sector_name = sector_relations[richest_sector[0]].name if richest_sector[0] < len(sector_relations) else f"Sector {richest_sector[0]}"
        print(f"Most capital-rich sector: {sector_name} (${richest_sector[1]:,.0f})")
    
    return env


def example_periodic_visualization():
    """Example 2: Run simulation with periodic visualization."""
    print("\n" + "=" * 60)
    print("Example 2: Periodic Visualization During Simulation")
    print("=" * 60)
    
    # Load config
    config = load_config("config/config.json")
    
    # Enable periodic visualization every 100 steps
    config.environment.visualize_every_n_steps = 100
    config.environment.visualization_dir = "visualizations/periodic"
    config.environment.save_plots = True
    config.environment.show_plots = False  # Don't block execution
    
    # Create environment
    env = IndustryEnv(config.environment)
    
    # Run simulation
    obs, info = env.reset(options={"initial_firms": 20})
    
    print(f"\nStarting simulation with {env.num_firms} firms")
    print(f"Visualizations will be saved every {env.visualize_every_n_steps} steps")
    
    for step in range(500):  # Run for 500 steps
        # Random actions for demonstration
        num_actions = np.random.randint(0, 3)
        action = {
            "num_actions": num_actions,
            "actions": []
        }
        
        for _ in range(num_actions):
            if np.random.random() < 0.5 and env.num_firms > 0:
                # Invest action
                firm_id = np.random.randint(0, env.num_firms)
                amount = np.random.uniform(1000, 10000)
                action["actions"].append({
                    "op": 0,
                    "invest": {"firm_id": firm_id, "amount": amount},
                    "create": {"sector": 0, "initial_capital": 0.0, "location": np.array([0.0, 0.0])}
                })
            else:
                # Create action
                sector = np.random.randint(0, env.num_sectors)
                capital = np.random.uniform(5000, 50000)
                location = np.array([
                    np.random.uniform(0, env.size),
                    np.random.uniform(0, env.size)
                ])
                action["actions"].append({
                    "op": 1,
                    "invest": {"firm_id": 0, "amount": 0.0},
                    "create": {"sector": sector, "initial_capital": capital, "location": location}
                })
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}: {env.num_firms} firms, Total Profit: {info['total_profit']:.2f}")
        
        if terminated or truncated:
            break
    
    print(f"\nSimulation completed. Check '{env.visualization_dir}' for visualization PNGs")
    
    return env


def example_save_environment(env: IndustryEnv):
    """Example 3: Save environment state."""
    print("\n" + "=" * 60)
    print("Example 3: Saving Environment State")
    print("=" * 60)
    
    # Save with automatic timestamp
    filepath = env.save_environment()
    print(f"\nEnvironment saved to: {filepath}")
    
    # Also save with custom name
    custom_path = "saved_envs/example_simulation.pkl"
    filepath = env.save_environment(custom_path)
    print(f"Environment also saved to: {filepath}")
    
    return filepath


def example_load_and_visualize(filepath: str):
    """Example 4: Load and visualize saved environment."""
    print("\n" + "=" * 60)
    print("Example 4: Loading and Visualizing Saved Environment")
    print("=" * 60)
    
    # Load the environment
    env = load_environment(filepath)
    print(f"\nEnvironment loaded from: {filepath}")
    print(f"  - Current step: {env.current_step}")
    print(f"  - Number of firms: {env.num_firms}")
    print(f"  - Total capital: {sum(c.capital for c in env.companies):,.2f}")
    
    # Create visualization
    print("\nCreating interactive visualization...")
    visualize_saved_environment(filepath, output_dir="visualizations/loaded")
    print("Interactive plot displayed. Close the window to continue.")


def example_manual_visualization():
    """Example 5: Manual visualization at specific points."""
    print("\n" + "=" * 60)
    print("Example 5: Manual Visualization Control")
    print("=" * 60)
    
    # Load config and disable automatic visualization
    config = load_config("config/config.json")
    config.environment.visualize_every_n_steps = 0  # Disable automatic
    config.environment.visualization_dir = "visualizations/manual"
    
    # Create environment
    env = IndustryEnv(config.environment)
    obs, info = env.reset(options={"initial_firms": 15})
    
    print("\nRunning simulation with manual visualization control...")
    
    # Run for a while
    for step in range(200):
        num_actions = np.random.randint(0, 3)
        action = {
            "num_actions": num_actions,
            "actions": []
        }
        
        for _ in range(num_actions):
            if np.random.random() < 0.5 and env.num_firms > 0:
                firm_id = np.random.randint(0, env.num_firms)
                amount = np.random.uniform(1000, 10000)
                action["actions"].append({
                    "op": 0,
                    "invest": {"firm_id": firm_id, "amount": amount},
                    "create": {"sector": 0, "initial_capital": 0.0, "location": np.array([0.0, 0.0])}
                })
            else:
                sector = np.random.randint(0, env.num_sectors)
                capital = np.random.uniform(5000, 50000)
                location = np.array([
                    np.random.uniform(0, env.size),
                    np.random.uniform(0, env.size)
                ])
                action["actions"].append({
                    "op": 1,
                    "invest": {"firm_id": 0, "amount": 0.0},
                    "create": {"sector": sector, "initial_capital": capital, "location": location}
                })
        
        obs, reward, terminated, truncated, info = env.step(action)
    
    # Manually trigger visualization at interesting points
    print("Creating manual visualization at step 200...")
    env.visualize_step("manual_checkpoint_200")
    
    # Run more steps
    for step in range(200, 400):
        num_actions = np.random.randint(0, 3)
        action = {"num_actions": num_actions, "actions": []}
        for _ in range(num_actions):
            if np.random.random() < 0.5 and env.num_firms > 0:
                firm_id = np.random.randint(0, env.num_firms)
                amount = np.random.uniform(1000, 10000)
                action["actions"].append({
                    "op": 0,
                    "invest": {"firm_id": firm_id, "amount": amount},
                    "create": {"sector": 0, "initial_capital": 0.0, "location": np.array([0.0, 0.0])}
                })
            else:
                sector = np.random.randint(0, env.num_sectors)
                capital = np.random.uniform(5000, 50000)
                location = np.array([np.random.uniform(0, env.size), np.random.uniform(0, env.size)])
                action["actions"].append({
                    "op": 1,
                    "invest": {"firm_id": 0, "amount": 0.0},
                    "create": {"sector": sector, "initial_capital": capital, "location": location}
                })
        obs, reward, terminated, truncated, info = env.step(action)
    
    print("Creating manual visualization at step 400...")
    env.visualize_step("manual_checkpoint_400")
    
    print(f"\nManual visualizations saved to: {env.visualization_dir}")


def main():
    """Main function with menu-driven interface."""
    print("=" * 70)
    print(" " * 15 + "INDUSTRY SIMULATION VISUALIZATION EXAMPLES")
    print("=" * 70)
    
    print("\nAvailable Examples:")
    print("  1. Basic Interactive Visualization - Quick demo with 4 plot types")
    print("  2. Periodic Visualization - Auto-save plots during long simulation")
    print("  3. Save & Load Environment - Persist and restore simulation state")
    print("  4. Manual Visualization - Control exactly when to create plots")
    print("  5. Run All Examples - Execute all demonstrations in sequence")
    print()
    
    choice = input("Select an example to run (1-5, or 'q' to quit): ").strip()
    
    if choice == '1':
        example_basic_visualization()
        
    elif choice == '2':
        env = example_periodic_visualization()
        print("\nWould you like to save this environment? (y/n): ", end='')
        if input().strip().lower() == 'y':
            example_save_environment(env)
            
    elif choice == '3':
        print("\nFirst, let's create and save an environment...")
        env = example_periodic_visualization()
        saved_path = example_save_environment(env)
        print("\nNow let's load and visualize it...")
        example_load_and_visualize(saved_path)
        
    elif choice == '4':
        example_manual_visualization()
        
    elif choice == '5':
        print("\nRunning all examples in sequence...\n")
        
        # Example 1: Basic visualization
        env = example_basic_visualization()
        
        # Example 2: Periodic visualization
        env = example_periodic_visualization()
        
        # Example 3: Save environment
        saved_path = example_save_environment(env)
        
        # Example 4: Load and visualize (commented out to avoid blocking)
        print("\n" + "=" * 60)
        print("Example 4: Load & Visualize (Skipped in 'Run All' mode)")
        print("=" * 60)
        print("To test loading, run Example 3 individually.")
        # example_load_and_visualize(saved_path)
        
        # Example 5: Manual visualization control
        example_manual_visualization()
        
        print("\n" + "=" * 70)
        print("All examples completed!")
        print("=" * 70)
        print("\nSummary of generated files:")
        print("  - Periodic visualizations: visualizations/periodic/")
        print("  - Saved environments: saved_envs/")
        print("  - Manual visualizations: visualizations/manual/")
        print("  - Loaded visualizations: visualizations/loaded/")
        print("\nUseful commands for working with saved environments:")
        print("  from env import load_environment, visualize_saved_environment")
        print("  env = load_environment('saved_envs/example_simulation.pkl')")
        print("  visualize_saved_environment('saved_envs/example_simulation.pkl')")
        
    elif choice.lower() == 'q':
        print("\nExiting. Goodbye!")
        return
        
    else:
        print("\nInvalid choice. Please run again and select 1-5 or 'q'.")
        return
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
