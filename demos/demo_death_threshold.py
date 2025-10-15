#!/usr/bin/env python3
"""Demo script showing company death threshold in action."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from env import IndustryEnv
import time

def demo_death_threshold():
    """Visual demonstration of company death mechanism."""
    
    print("=" * 70)
    print("COMPANY DEATH THRESHOLD DEMO")
    print("=" * 70)
    print("\nThis demo shows how companies die when capital falls below threshold.\n")
    
    # Create environment with moderate threshold
    death_threshold = 10000.0
    config = Config()
    config.environment.death_threshold = death_threshold
    config.environment.logistic_cost_rate = 500.0  # High costs to trigger deaths
    env = IndustryEnv(config.environment)
    
    obs, _ = env.reset(options={"initial_firms": 20})
    
    print(f"🏭 Starting simulation with {env.num_firms} companies")
    print(f"💀 Death threshold: ${death_threshold:,.0f}")
    print(f"📊 High logistic costs to create competitive pressure\n")
    print("=" * 70)
    
    # Show initial state
    print("\n📈 INITIAL STATE:")
    capitals = sorted([c.capital for c in env.companies], reverse=True)
    print(f"   Total companies: {len(capitals)}")
    print(f"   Capital range: ${min(capitals):,.2f} - ${max(capitals):,.2f}")
    print(f"   Average capital: ${sum(capitals)/len(capitals):,.2f}")
    at_risk = sum(1 for c in capitals if c < death_threshold * 1.5)
    print(f"   At risk (< ${death_threshold*1.5:,.0f}): {at_risk}")
    
    # Run simulation
    print("\n" + "=" * 70)
    print("SIMULATION RUNNING...")
    print("=" * 70 + "\n")
    
    total_deaths = 0
    step_deaths = []
    
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        num_deaths = info.get('num_deaths', 0)
        total_deaths += num_deaths
        step_deaths.append(num_deaths)
        
        if num_deaths > 0:
            print(f"💀 Step {step+1:2d}: {num_deaths} companies died | "
                  f"{info['num_firms']} surviving | "
                  f"Profit: ${info['total_profit']:,.2f}")
        elif step % 10 == 0:
            print(f"✓  Step {step+1:2d}: All alive | "
                  f"{info['num_firms']} companies | "
                  f"Profit: ${info['total_profit']:,.2f}")
    
    # Final state
    print("\n" + "=" * 70)
    print("📊 FINAL STATE:")
    print("=" * 70)
    
    if env.companies:
        capitals = sorted([c.capital for c in env.companies], reverse=True)
        print(f"\n   Surviving companies: {len(capitals)}")
        print(f"   Capital range: ${min(capitals):,.2f} - ${max(capitals):,.2f}")
        print(f"   Average capital: ${sum(capitals)/len(capitals):,.2f}")
        print(f"   Total deaths: {total_deaths}")
        print(f"   Survival rate: {len(capitals)/20*100:.1f}%")
        
        # Show capital distribution
        print("\n   Top 5 companies by capital:")
        for i, cap in enumerate(capitals[:5], 1):
            print(f"      {i}. ${cap:,.2f}")
    else:
        print("\n   💀 ALL COMPANIES DIED!")
        print(f"   Total deaths: {total_deaths}")
    
    # Death statistics
    print("\n" + "=" * 70)
    print("📈 DEATH STATISTICS:")
    print("=" * 70)
    
    steps_with_deaths = sum(1 for d in step_deaths if d > 0)
    print(f"\n   Steps with deaths: {steps_with_deaths}/50")
    print(f"   Average deaths per deadly step: {total_deaths/max(steps_with_deaths, 1):.2f}")
    print(f"   Max deaths in single step: {max(step_deaths)}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("💡 INTERPRETATION:")
    print("=" * 70)
    print(f"""
The death threshold acts as a bankruptcy mechanism:

1. Companies with capital < ${death_threshold:,.0f} are removed
2. High logistic costs (${500.0}) create financial pressure
3. Weak companies fail, strong companies survive
4. This creates natural selection in the economy

Effect on RL training:
- Agent must consider long-term viability
- Poor location choices lead to company death
- Encourages strategic investment decisions
- Creates dynamic, realistic economic environment
    """)
    
    # Recommendations
    print("=" * 70)
    print("⚙️  RECOMMENDED SETTINGS:")
    print("=" * 70)
    print("""
For different scenarios:

1. No deaths (study full dynamics):
   death_threshold = 0.0
   
2. Realistic business (moderate pressure):
   death_threshold = 5000.0 - 10000.0
   
3. Harsh economy (high pressure):
   death_threshold = 20000.0 - 50000.0
   
4. Survival of fittest only:
   death_threshold = initial_capital_max (e.g., 100000.0)

Combine with other parameters:
- High logistic_cost_rate → More deaths (location matters)
- Low revenue_rate → More deaths (low margins)
- High op_cost_rate → More deaths (expensive operations)
    """)

if __name__ == "__main__":
    demo_death_threshold()
