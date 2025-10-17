import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import Config
from config.config import load_config
from env import IndustryEnv
from env.sector import sector_relations
from tqdm import tqdm


def run_and_visualize(steps: int = 400, initial_firms: int = 150, save_every: int = 20, out_dir: str = "visualizations/periodic"):
    config = load_config("config/config.json")
    # Use config.json settings (no hardcoded overrides)

    env = IndustryEnv(config.environment)
    obs, _ = env.reset(seed=42, options={"initial_firms": initial_firms})

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Track statistics over time
    stats_history = []

    for step in tqdm(range(1, steps + 1)):
        # Step environment with zero action (no agent intervention)
        zero_action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(zero_action)
        
        env.current_step = step
        
        # Collect statistics
        sector_counts = {}
        sector_capitals = {}
        for company in env.companies:
            sector_name = sector_relations[company.sector_id].name
            sector_counts[sector_name] = sector_counts.get(sector_name, 0) + 1
            sector_capitals[sector_name] = sector_capitals.get(sector_name, 0.0) + company.capital
        
        total_capital = sum(c.capital for c in env.companies)
        stats_history.append({
            'step': step,
            'total_companies': len(env.companies),
            'total_capital': total_capital,
            'sector_counts': sector_counts.copy(),
            'sector_capitals': sector_capitals.copy()
        })

        if step % save_every == 0:
            # Always draw a simple scatter plot as requested
            fig, ax = plt.subplots(figsize=(10, 7))
            xs = [c.location[0] for c in env.companies]
            ys = [c.location[1] for c in env.companies]
            caps = [max(c.capital, 0.0) for c in env.companies]
            sizes = [max(min(c.capital, 5_000.0) / 50.0, 10.0) for c in env.companies]  # size by capital

            if len(xs) > 0:
                sc = ax.scatter(xs, ys, c=caps, cmap='viridis', s=sizes, alpha=0.8, edgecolors='none')
                fig.colorbar(sc, ax=ax, label='Capital')
            else:
                ax.scatter([], [])

            # Axes and title
            size = getattr(env, 'size', 100.0)
            ax.set_xlim(0, size)
            ax.set_ylim(0, size)
            ax.set_aspect('equal', adjustable='box')

            alive = len(env.companies)
            total_cap = sum(c.capital for c in env.companies)
            ax.set_title(f"Step {step} | Alive: {alive} | Total Capital: {total_cap:,.0f}")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.2)

            # Add sector type labels for each company
            sector_abbrev = {
                "Raw": "R",
                "Parts": "P",
                "Electronics": "E",
                "Battery/Motor": "B",
                "OEM": "O",
                "Service": "S",
                "Other": "X",
            }
            
            for company in env.companies:
                try:
                    sector_name = sector_relations[company.sector_id].name
                    abbrev = sector_abbrev.get(sector_name, "?")
                    ax.text(company.location[0], company.location[1], abbrev, 
                            fontsize=7, ha='center', va='center', color='black', weight='normal',
                            bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.8, pad=0.2),
                            zorder=10, clip_on=True)
                except Exception as e:
                    print(f"Error adding label for company at {company.location}: {e}")

            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"step_{step:06d}.png"), dpi=150)
            plt.close(fig)
        
        # Check if episode ended early
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    # Print final statistics
    print("\n" + "="*80)
    print("SIMULATION SUMMARY")
    print("="*80)
    
    # Print statistics at key milestones
    milestones = [10, 50, 100, 200, 300, 400, 500]
    for milestone in milestones:
        if milestone <= len(stats_history):
            stat = stats_history[milestone - 1]
            print(f"\n--- Step {stat['step']} ---")
            print(f"Total Companies: {stat['total_companies']}")
            print(f"Total Capital: {stat['total_capital']:,.0f}")
            print(f"Sector Distribution:")
            for sector in ["Raw", "Parts", "Electronics", "Battery/Motor", "OEM", "Service", "Other"]:
                count = stat['sector_counts'].get(sector, 0)
                capital = stat['sector_capitals'].get(sector, 0.0)
                if count > 0:
                    print(f"  {sector:15s}: {count:3d} companies, {capital:12,.0f} capital (avg: {capital/count:,.0f})")
    
    # Print trend analysis
    print("\n" + "="*80)
    print("TREND ANALYSIS")
    print("="*80)
    
    if len(stats_history) >= 2:
        start = stats_history[0]
        end = stats_history[-1]
        
        print(f"\nOverall Change (Step {start['step']} → {end['step']}):")
        print(f"  Total Companies: {start['total_companies']} → {end['total_companies']} ({end['total_companies'] - start['total_companies']:+d})")
        print(f"  Total Capital: {start['total_capital']:,.0f} → {end['total_capital']:,.0f} ({end['total_capital'] - start['total_capital']:+,.0f})")
        
        print(f"\nSector Survival Rates:")
        for sector in ["Raw", "Parts", "Electronics", "Battery/Motor", "OEM", "Service", "Other"]:
            start_count = start['sector_counts'].get(sector, 0)
            end_count = end['sector_counts'].get(sector, 0)
            if start_count > 0:
                survival_rate = (end_count / start_count) * 100
                print(f"  {sector:15s}: {start_count:3d} → {end_count:3d} ({survival_rate:5.1f}% survival)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    run_and_visualize()


