"""
Visual demonstration of logistic costs in action.
Creates a scenario with companies at different distances and shows cost impacts.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from config import Config
from env import Company, sector_relations, IndustryEnv

def visualize_cost_vs_distance():
    """Create a plot showing how logistic cost varies with distance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Cost vs Distance for different cost rates
    distances = np.linspace(1, 50, 49)
    trade_volume = 1000.0
    
    for rate in [0.1, 1.0, 10.0, 100.0]:
        costs = [rate * trade_volume * d for d in distances]
        ax1.plot(distances, costs, label=f'k = {rate}', linewidth=2)
    
    ax1.set_xlabel('Distance', fontsize=12)
    ax1.set_ylabel('Logistic Cost ($)', fontsize=12)
    ax1.set_title('Logistic Cost vs Distance\n(Inverse Square Law)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=1)
    
    # Plot 2: Net profit vs Distance
    revenue = 50000.0
    distance_range = np.linspace(1, 50, 50)
    
    for rate in [0.01, 0.1]:
        net_profits = [revenue - (rate * revenue * 0.2 * d) for d in distance_range]
        ax2.plot(distance_range, net_profits, label=f'k = {rate}', linewidth=2, marker='o', markersize=3)
    
    ax2.axhline(y=revenue, color='green', linestyle='--', label='Max Profit (no logistics)', alpha=0.5)
    ax2.set_xlabel('Distance to Customer', fontsize=12)
    ax2.set_ylabel('Net Profit ($)', fontsize=12)
    ax2.set_title('Company Profitability vs Customer Distance', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demos/demo_logistic_cost_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: demos/demo_logistic_cost_analysis.png")
    plt.show()

def simulate_clustering_benefit():
    """Demonstrate how companies benefit from clustering near supply chain partners."""
    print("\n" + "="*70)
    print("CLUSTERING BENEFIT SIMULATION")
    print("="*70)
    print("\n配置说明：")
    print("  - 3个Tier 0公司（Raw材料）提供充足供应")
    print("  - 7个下游公司从上游购买并产生物流成本")
    print("  - production_capacity_ratio=0.5 确保供应充足")
    print("  - logistic_cost_rate=100.0 使成本更明显")
    print("  - 运行5个时间步以累积物流成本\n")
    
    # Scenario 1: Dispersed companies
    print("\n📍 Scenario 1: DISPERSED LAYOUT")
    print("   Companies spread across the map\n")
    
    config1 = Config()
    config1.environment.logistic_cost_rate = 100.0  # 提高费率使成本更明显
    # 增加生产能力以确保供应充足
    config1.environment.production_capacity_ratio = 0.5  # 从0.1提高到0.5
    env1 = IndustryEnv(config1.environment)
    env1.reset()
    
    # Create dispersed companies
    # 使用更多会产生物流成本的sector（避免过多Tier 0和Other sector）
    locations_dispersed = [
        (10, 10), (90, 90), (10, 90), (90, 10), (50, 50),
        (25, 75), (75, 25), (30, 50), (70, 50), (50, 30)
    ]
    
    # 使用sector_id: [0, 0, 0, 1, 2, 3, 4, 5, 1, 2]
    # 3个Tier 0公司（Raw）提供充足供应，7个下游公司会购买并产生物流成本
    sector_ids = [0, 0, 0, 1, 2, 3, 4, 5, 1, 2]
    
    for i, loc in enumerate(locations_dispersed):
        sector_id = sector_ids[i]
        env1.companies.append(Company(100000, sector_id, loc, logistic_cost_rate=1.0))
        env1.num_firms += 1
    
    # Build supply chain network and run simulation multiple times
    # 运行多次以累积物流成本
    if env1.enable_products:
        env1._build_supply_chain_network()
        
        # 运行5个时间步来累积物流成本
        for step in range(5):
            env1._simulate_supply_chain()
    
    total_cost_dispersed = sum(c.logistic_cost for c in env1.companies)
    avg_cost_dispersed = total_cost_dispersed / len(env1.companies) if env1.companies else 0
    
    print(f"   Total logistic cost: ${total_cost_dispersed:,.2f}")
    print(f"   Avg cost per company: ${avg_cost_dispersed:,.2f}")
    
    # 显示每个公司的详细信息
    non_zero_costs = sum(1 for c in env1.companies if c.logistic_cost > 0)
    print(f"   Companies with logistic costs: {non_zero_costs}/{len(env1.companies)}")
    
    # Scenario 2: Clustered companies
    print("\n📍 Scenario 2: CLUSTERED LAYOUT")
    print("   Companies grouped by supply chain relationships\n")
    
    config2 = Config()
    config2.environment.logistic_cost_rate = 100.0  # 提高费率使成本更明显
    # 增加生产能力以确保供应充足
    config2.environment.production_capacity_ratio = 0.5  # 从0.1提高到0.5
    env2 = IndustryEnv(config2.environment)
    env2.reset()
    
    # Create clustered companies (sectors 0-2 together, 3-5 together, etc.)
    locations_clustered = [
        (20, 20), (22, 22), (24, 20), (22, 18), (20, 24),  # Cluster 1
        (50, 50), (52, 50), (50, 52), (48, 50), (50, 48)   # Cluster 2
    ]
    
    # 使用相同的sector分配以便公平比较
    # 3个Tier 0公司（Raw）提供充足供应，7个下游公司会购买并产生物流成本
    sector_ids = [0, 0, 0, 1, 2, 3, 4, 5, 1, 2]
    
    for i, loc in enumerate(locations_clustered):
        sector_id = sector_ids[i]
        env2.companies.append(Company(100000, sector_id, loc, logistic_cost_rate=1.0))
        env2.num_firms += 1
    
    # Build supply chain network and run simulation multiple times
    # 运行多次以累积物流成本
    if env2.enable_products:
        env2._build_supply_chain_network()
        
        # 运行5个时间步来累积物流成本
        for step in range(5):
            env2._simulate_supply_chain()
    
    total_cost_clustered = sum(c.logistic_cost for c in env2.companies)
    avg_cost_clustered = total_cost_clustered / len(env2.companies) if env2.companies else 0
    
    print(f"   Total logistic cost: ${total_cost_clustered:,.2f}")
    print(f"   Avg cost per company: ${avg_cost_clustered:,.2f}")
    
    # 显示每个公司的详细信息
    non_zero_costs = sum(1 for c in env2.companies if c.logistic_cost > 0)
    print(f"   Companies with logistic costs: {non_zero_costs}/{len(env2.companies)}")
    
    # Comparison
    print("\n📊 COMPARISON")
    print("   " + "-"*50)
    if total_cost_dispersed > 0:
        cost_reduction = ((total_cost_dispersed - total_cost_clustered) / total_cost_dispersed) * 100
        print(f"   Cost reduction from clustering: {cost_reduction:.1f}%")
        print(f"   Savings: ${total_cost_dispersed - total_cost_clustered:,.2f}")
    else:
        print("   Note: No logistic costs incurred (product system may be disabled)")
        cost_reduction = 0
    
    # Visualize both scenarios
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot dispersed
    from matplotlib.cm import get_cmap
    tab10 = get_cmap('tab10')
    for i, company in enumerate(env1.companies):
        color = tab10(company.sector_id % 7)
        size = 200 + company.logistic_cost / 100
        ax1.scatter(company.x, company.y, c=[color], s=size, alpha=0.6, edgecolors='black')
        ax1.text(company.x, company.y+3, f'${company.logistic_cost:.0f}', 
                ha='center', fontsize=8, fontweight='bold')
    
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title(f'Dispersed Layout\nTotal Cost: ${total_cost_dispersed:,.0f}', 
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot clustered
    for i, company in enumerate(env2.companies):
        color = tab10(company.sector_id % 7)
        size = 200 + company.logistic_cost / 100
        ax2.scatter(company.x, company.y, c=[color], s=size, alpha=0.6, edgecolors='black')
        ax2.text(company.x, company.y+2, f'${company.logistic_cost:.0f}', 
                ha='center', fontsize=8, fontweight='bold')
    
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title(f'Clustered Layout\nTotal Cost: ${total_cost_clustered:,.0f} ({cost_reduction:.1f}% lower)', 
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demos/demo_clustering_benefit.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved: demos/demo_clustering_benefit.png")
    plt.show()

def interactive_distance_calculator():
    """Interactive tool to calculate costs at different distances."""
    print("\n" + "="*70)
    print("INTERACTIVE LOGISTIC COST CALCULATOR")
    print("="*70)
    
    company = Company(100000, 1, (0, 0), logistic_cost_rate=100.0)
    
    print("\nCalculate logistic costs for different scenarios:")
    print("(Supplier at location (0, 0), logistic_cost_rate = 100.0)\n")
    
    scenarios = [
        ("Nearby customer", 5.0, 1000.0),
        ("Medium distance", 20.0, 2000.0),
        ("Far customer", 50.0, 1500.0),
        ("Very close partner", 2.0, 500.0),
        ("Distant market", 80.0, 3000.0),
    ]
    
    print(f"{'Scenario':<25} | {'Distance':>10} | {'Volume':>12} | {'Cost':>12} | {'Cost/Unit':>12}")
    print("-" * 80)
    
    for name, distance, volume in scenarios:
        customer = Company(100000, 2, (distance, 0))
        cost = company.calculate_logistic_cost_to(customer, volume)
        cost_per_unit = cost / volume
        
        print(f"{name:<25} | {distance:>10.1f} | ${volume:>11.2f} | ${cost:>11.2f} | ${cost_per_unit:>11.4f}")
    
    print("\n💡 Insight: Cost per unit decreases dramatically with distance!")
    print("   This incentivizes supply chain clustering in the simulation.")

if __name__ == "__main__":
    print("="*70)
    print("LOGISTIC COST VISUALIZATION DEMO")
    print("="*70)
    print("\nThis demo shows how distance-based logistic costs affect companies")
    print("and why clustering near supply chain partners is beneficial.\n")
    
    # Run demonstrations
    visualize_cost_vs_distance()
    simulate_clustering_benefit()
    interactive_distance_calculator()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key Takeaways:

1. 📉 Inverse Square Law: Costs decrease rapidly with distance
   - Doubling distance → 75% cost reduction
   - Very distant partners have minimal per-unit costs

2. 🏘️  Clustering Benefits: Companies near supply chain partners save money
   - Lower logistic costs
   - Higher net profitability
   - Better capital accumulation

3. 🎯 Strategic Implications for RL:
   - Location decisions now have real economic consequences
   - Agents must balance accessibility vs. cost efficiency
   - Supply chain clustering becomes an emergent strategy

4. ⚙️  Configurable: The logistic_cost_rate hyperparameter controls magnitude
   - Low rates (10-50): Distance matters less
   - Medium rates (100-500): Balanced gameplay
   - High rates (1000+): Strong clustering incentive

The logistic cost system adds spatial economic depth to the simulation!
""")
