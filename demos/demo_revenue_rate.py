#!/usr/bin/env python3
"""Demo script showing revenue_rate hyperparameter in action."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from env import Company, IndustryEnv

def demo_revenue_rate():
    """Demonstrate how revenue_rate affects company revenues in simulation."""
    
    print("=" * 70)
    print("REVENUE RATE HYPERPARAMETER DEMO")
    print("=" * 70)
    
    print("\nFormula: revenue = revenue_rate × order_amount")
    print("This hyperparameter controls how efficiently companies convert")
    print("orders/trades into revenue.\n")
    
    # Scenario 1: Normal revenue rate (1.0)
    print("\n" + "=" * 70)
    print("SCENARIO 1: Normal Revenue Rate (1.0)")
    print("=" * 70)
    
    config1 = Config()
    config1.environment.revenue_rate = 1.0
    config1.environment.logistic_cost_rate = 100.0
    env1 = IndustryEnv(config1.environment)
    obs1, _ = env1.reset(options={"initial_firms": 5})
    
    # Simulate some steps and track metrics
    total_profit = 0
    for _ in range(10):
        action = env1.action_space.sample()
        obs1, reward, done, truncated, info = env1.step(action)
        total_profit += info.get('total_profit', 0)
    
    print(f"Companies created: {len(env1.companies)}")
    if env1.companies:
        total_orders = sum(c.orders for c in env1.companies)
        total_capital = sum(c.capital for c in env1.companies)
        print(f"Total orders processed: {total_orders}")
        print(f"Total capital: ${total_capital:.2f}")
        print(f"Total profit (10 steps): ${total_profit:.2f}")
        if total_orders > 0:
            print(f"Average profit per order: ${total_profit/total_orders:.2f}")
    
    # Scenario 2: High revenue rate (2.0)
    print("\n" + "=" * 70)
    print("SCENARIO 2: High Revenue Rate (2.0)")
    print("=" * 70)
    print("Companies are twice as efficient at converting orders to revenue\n")
    
    config2 = Config()
    config2.environment.revenue_rate = 2.0
    config2.environment.logistic_cost_rate = 100.0
    env2 = IndustryEnv(config2.environment)
    obs2, _ = env2.reset(options={"initial_firms": 5})
    
    # Simulate same number of steps and track metrics
    total_profit = 0
    for _ in range(10):
        action = env2.action_space.sample()
        obs2, reward, done, truncated, info = env2.step(action)
        total_profit += info.get('total_profit', 0)
    
    print(f"Companies created: {len(env2.companies)}")
    if env2.companies:
        total_orders = sum(c.orders for c in env2.companies)
        total_capital = sum(c.capital for c in env2.companies)
        print(f"Total orders processed: {total_orders}")
        print(f"Total capital: ${total_capital:.2f}")
        print(f"Total profit (10 steps): ${total_profit:.2f}")
        if total_orders > 0:
            print(f"Average profit per order: ${total_profit/total_orders:.2f}")
    
    # Scenario 3: Low revenue rate (0.5)
    print("\n" + "=" * 70)
    print("SCENARIO 3: Low Revenue Rate (0.5)")
    print("=" * 70)
    print("Companies are less efficient (50% conversion rate)\n")
    
    config3 = Config()
    config3.environment.revenue_rate = 0.5
    config3.environment.logistic_cost_rate = 100.0
    env3 = IndustryEnv(config3.environment)
    obs3, _ = env3.reset(options={"initial_firms": 5})
    
    # Simulate same number of steps and track metrics
    total_profit = 0
    for _ in range(10):
        action = env3.action_space.sample()
        obs3, reward, done, truncated, info = env3.step(action)
        total_profit += info.get('total_profit', 0)
    
    print(f"Companies created: {len(env3.companies)}")
    if env3.companies:
        total_orders = sum(c.orders for c in env3.companies)
        total_capital = sum(c.capital for c in env3.companies)
        print(f"Total orders processed: {total_orders}")
        print(f"Total capital: ${total_capital:.2f}")
        print(f"Total profit (10 steps): ${total_profit:.2f}")
        if total_orders > 0:
            print(f"Average profit per order: ${total_profit/total_orders:.2f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("USE CASES")
    print("=" * 70)
    print("""
1. revenue_rate = 1.0 (default)
   → Standard 1:1 conversion of orders to revenue
   
2. revenue_rate > 1.0
   → High-margin industries (tech, finance)
   → Companies generate more revenue per order
   → Encourages aggressive expansion
   
3. revenue_rate < 1.0
   → Low-margin industries (retail, commodities)
   → Companies need more orders for same revenue
   → Encourages efficiency and scale
   
4. Sector-specific rates (future enhancement)
   → Different revenue_rate per industry sector
   → Model realistic profit margins
    """)
    
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print("""
Edit config.json:

  "supply_chain": {
    "trade_volume_fraction": 0.01,
    "revenue_rate": 1.0,        ← Adjust this value
    "enable_supply_chain": true,
    "min_distance_epsilon": 0.1
  }

Or pass to IndustryEnv:

  env = IndustryEnv(revenue_rate=2.0)
    """)

if __name__ == "__main__":
    demo_revenue_rate()
