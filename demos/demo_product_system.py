"""
Demo script showing the product system in action.

This demonstrates:
1. How upstream companies (tier 0) produce products
2. How downstream companies purchase from upstream
3. Product inventory tracking
4. Supply chain network formation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from env import IndustryEnv, sector_relations

def demo_product_system():
    """Demonstrate the product system with a simple scenario."""
    
    print("=" * 80)
    print("PRODUCT SYSTEM DEMO")
    print("=" * 80)
    
    # Load config with product system enabled
    config = Config()
    config.environment.enable_products = True
    config.environment.production_capacity_ratio = 0.1
    config.environment.purchase_budget_ratio = 0.2
    config.environment.initial_firms = 0  # Start with no firms, add manually
    
    print("\n📋 Configuration:")
    print(f"   Product system enabled: {config.environment.enable_products}")
    print(f"   Production capacity ratio: {config.environment.production_capacity_ratio}")
    print(f"   Purchase budget ratio: {config.environment.purchase_budget_ratio}")
    
    # Create environment
    env = IndustryEnv(config.environment)
    obs, info = env.reset(options={"initial_firms": 0})
    
    print("\n🏭 Creating companies in different tiers...")
    
    # Create companies in different tiers for demonstration
    # Current automotive supply chain (7 sectors):
    # Tier 0: Raw Materials (sector 0)
    # Tier 1: Parts Manufacturers (sector 1), Electronics (sector 2), Battery/Motor (sector 3)
    # Tier 2: OEM Assembly (sector 4)
    # Tier 3: Service Providers (sector 5)
    # Tier 4: Other (sector 6) - isolated
    
    companies_to_create = [
        # (sector_id, capital, description)
        (0, 50000, f"{sector_relations[0].name} (Tier 0)"),
        (1, 60000, f"{sector_relations[1].name} (Tier 1)"),
        (2, 40000, f"{sector_relations[2].name} (Tier 1)"),
        (3, 35000, f"{sector_relations[3].name} (Tier 1)"),
        (4, 30000, f"{sector_relations[4].name} (Tier 2)"),
        (5, 25000, f"{sector_relations[5].name} (Tier 3)"),
    ]
    
    import numpy as np
    for idx, (sector_id, capital, description) in enumerate(companies_to_create):
        # Create action to add company - use sample and modify
        sampled_action = env.action_space.sample()
        # Create a proper action dict
        action = {
            'num_actions': 1,
            'actions': []
        }
        # Vary locations to prevent clustering
        location = [40.0 + idx * 10.0, 40.0 + idx * 5.0]
        # Add the create action
        for i in range(env.max_actions_per_step):
            if i == 0:
                action['actions'].append({
                    "op": 1,  # Create
                    "invest": {"firm_id": 0, "amount": [0.0]},
                    "create": {
                        "initial_capital": [float(capital)],
                        "sector": sector_id,
                        "location": location
                    }
                })
            else:
                action['actions'].append(sampled_action['actions'][i])
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   ✓ Created {description} with capital ${capital:,}")
        print(f"      Total companies now: {len(env.companies)}")
    
    print(f"\n✅ Created {env.num_firms} companies")
    
    # Display supply chain network
    print("\n🔗 Supply Chain Network:")
    for i, company in enumerate(env.companies):
        sector = sector_relations[company.sector_id]
        sector_name = sector.name
        print(f"\n   Company {i}: {sector_name} (Tier {company.tier})")
        print(f"      Capital: ${company.capital:,.2f}")
        print(f"      Max Production: {company.get_max_production():,.2f}")
        print(f"      Max Purchase Budget: ${company.get_max_purchase_budget():,.2f}")
        print(f"      Suppliers: {len(company.suppliers)} companies")
        print(f"      Customers: {len(company.customers)} companies")
    
    # Run simulation for a few steps
    print("\n" + "=" * 80)
    print("RUNNING SIMULATION")
    print("=" * 80)
    
    for step in range(5):
        print(f"\n📅 Step {step + 1}:")
        
        # No actions this step, just observe supply chain
        sampled_action = env.action_space.sample()
        action = {
            'num_actions': 0,
            'actions': sampled_action['actions']
        }
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"   Total produced: {info['total_produced']:,.2f}")
        print(f"   Total purchased: {info['total_purchased']:,.2f}")
        print(f"   Total sold: {info['total_sold']:,.2f}")
        print(f"   Total inventory: {info['total_inventory']:,.2f}")
        print(f"   Total profit: ${info['total_profit']:,.2f}")
        
        # Show individual company status
        print(f"\n   Company Details:")
        for i, company in enumerate(env.companies):
            print(f"      Company {i} (Tier {company.tier}):")
            print(f"         Capital: ${company.capital:,.2f}")
            print(f"         Inventory: {company.product_inventory:,.2f}")
            print(f"         Produced: {company.products_produced_this_step:,.2f}")
            print(f"         Purchased: {company.products_purchased_this_step:,.2f}")
            print(f"         Sold: {company.products_sold_this_step:,.2f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\n✅ Product system demonstration complete!")
    print("\n📊 Key Observations:")
    print("   1. Tier 0 companies produce products from scratch")
    print("   2. Higher tier companies purchase from lower tiers")
    print("   3. Each company's production/purchase is limited by capital")
    print("   4. Inventory flows through the supply chain")
    print("   5. Revenue is generated when products are sold to downstream customers")
    
    return env

if __name__ == "__main__":
    env = demo_product_system()
