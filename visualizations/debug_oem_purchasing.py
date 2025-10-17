"""
Debug script to trace OEM purchasing behavior.
Check if OEM is actually buying from P/E/B suppliers.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.env import IndustryEnv
from config.config import Config
from env.sector import sector_relations
import numpy as np


def trace_oem_purchases():
    """Trace OEM purchasing in detail."""
    print("=" * 80)
    print("OEM PURCHASING BEHAVIOR TRACE")
    print("=" * 80)
    
    config = Config.from_json("config/config.json")
    env = IndustryEnv(config.environment)
    obs, info = env.reset(options={"initial_firms": 30})
    
    print(f"\n🏭 Initial State: {len(env.companies)} companies")
    
    # Identify companies
    oem_companies = [c for c in env.companies if sector_relations[c.sector_id].name == "OEM"]
    parts_companies = [c for c in env.companies if sector_relations[c.sector_id].name == "Parts"]
    elec_companies = [c for c in env.companies if sector_relations[c.sector_id].name == "Electronics"]
    batt_companies = [c for c in env.companies if sector_relations[c.sector_id].name == "Battery/Motor"]
    
    print(f"   OEM: {len(oem_companies)}")
    print(f"   Parts: {len(parts_companies)}")
    print(f"   Electronics: {len(elec_companies)}")
    print(f"   Battery/Motor: {len(batt_companies)}")
    
    # Track for 5 steps
    for step in range(1, 6):
        print(f"\n{'=' * 80}")
        print(f"STEP {step}")
        print('=' * 80)
        
        # Take zero action
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
        
        # Before step - check OEM state
        oem_companies = [c for c in env.companies if sector_relations[c.sector_id].name == "OEM"]
        
        if not oem_companies:
            print("\n❌ NO OEM COMPANIES LEFT!")
            break
        
        print(f"\n🏭 OEM Companies ({len(oem_companies)} alive):")
        
        for i, oem in enumerate(oem_companies[:3]):
            print(f"\n   OEM #{i+1}:")
            print(f"      Capital: ${oem.capital:,.0f}")
            print(f"      Purchase Budget: ${oem.get_max_purchase_budget():,.0f}")
            print(f"      Suppliers: {len(oem.suppliers)}")
            
            # Check supplier breakdown
            parts_suppliers = [s for s in oem.suppliers if sector_relations[s.sector_id].name == "Parts"]
            elec_suppliers = [s for s in oem.suppliers if sector_relations[s.sector_id].name == "Electronics"]
            batt_suppliers = [s for s in oem.suppliers if sector_relations[s.sector_id].name == "Battery/Motor"]
            
            print(f"         Parts suppliers: {len(parts_suppliers)}")
            print(f"         Electronics suppliers: {len(elec_suppliers)}")
            print(f"         Battery suppliers: {len(batt_suppliers)}")
            
            # Current inventory
            print(f"      Current Inventory:")
            print(f"         Parts: {oem.parts_inventory:.2f}")
            print(f"         Electronics: {oem.electronics_inventory:.2f}")
            print(f"         Battery: {oem.battery_inventory:.2f}")
            print(f"         Product (OEM): {oem.product_inventory:.2f}")
        
        # Execute step
        obs, reward, terminated, truncated, info = env.step(zero_action)
        
        # After step - check what was purchased
        oem_companies = [c for c in env.companies if sector_relations[c.sector_id].name == "OEM"]
        
        print(f"\n📊 After Step {step}:")
        
        for i, oem in enumerate(oem_companies[:3]):
            print(f"\n   OEM #{i+1}:")
            print(f"      Purchased: {oem.products_purchased_this_step:.2f} units")
            print(f"      Produced: {oem.products_produced_this_step:.2f} OEM units")
            print(f"      Sold: {oem.products_sold_this_step:.2f} OEM units")
            print(f"      New Inventory:")
            print(f"         Parts: {oem.parts_inventory:.2f}")
            print(f"         Electronics: {oem.electronics_inventory:.2f}")
            print(f"         Battery: {oem.battery_inventory:.2f}")
            print(f"         Product (OEM): {oem.product_inventory:.2f}")
            print(f"      Capital: ${oem.capital:,.0f}")
        
        # Check if P/E/B got any revenue
        parts_companies = [c for c in env.companies if sector_relations[c.sector_id].name == "Parts"]
        elec_companies = [c for c in env.companies if sector_relations[c.sector_id].name == "Electronics"]
        batt_companies = [c for c in env.companies if sector_relations[c.sector_id].name == "Battery/Motor"]
        
        print(f"\n💰 P/E/B Revenue Check:")
        
        if parts_companies:
            total_sales = sum([c.products_sold_this_step for c in parts_companies])
            total_prod = sum([c.products_produced_this_step for c in parts_companies])
            print(f"   Parts: {len(parts_companies)} alive, sold {total_sales:.2f}, produced {total_prod:.2f}")
        else:
            print(f"   Parts: ALL DEAD")
        
        if elec_companies:
            total_sales = sum([c.products_sold_this_step for c in elec_companies])
            total_prod = sum([c.products_produced_this_step for c in elec_companies])
            print(f"   Electronics: {len(elec_companies)} alive, sold {total_sales:.2f}, produced {total_prod:.2f}")
        else:
            print(f"   Electronics: ALL DEAD")
        
        if batt_companies:
            total_sales = sum([c.products_sold_this_step for c in batt_companies])
            total_prod = sum([c.products_produced_this_step for c in batt_companies])
            print(f"   Battery: {len(batt_companies)} alive, sold {total_sales:.2f}, produced {total_prod:.2f}")
        else:
            print(f"   Battery: ALL DEAD")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    
    oem_companies = [c for c in env.companies if sector_relations[c.sector_id].name == "OEM"]
    
    if oem_companies:
        sample_oem = oem_companies[0]
        budget = sample_oem.get_max_purchase_budget()
        print(f"\n📊 Sample OEM Analysis:")
        print(f"   Capital: ${sample_oem.capital:,.0f}")
        print(f"   Purchase Budget Ratio: {config.environment.purchase_budget_ratio}")
        print(f"   Max Purchase Budget: ${budget:,.0f}")
        print(f"   Suppliers: {len(sample_oem.suppliers)}")
        
        if budget > 0:
            print(f"\n   ✓ OEM has budget to purchase")
        else:
            print(f"\n   ❌ OEM has NO budget to purchase!")
        
        if sample_oem.suppliers:
            print(f"   ✓ OEM has suppliers")
            # Check if suppliers are alive and have inventory
            alive_suppliers = [s for s in sample_oem.suppliers if s in env.companies]
            print(f"   ✓ {len(alive_suppliers)}/{len(sample_oem.suppliers)} suppliers are alive")
            
            suppliers_with_inventory = [s for s in alive_suppliers if s.product_inventory > 0]
            print(f"   ✓ {len(suppliers_with_inventory)} suppliers have inventory")
        else:
            print(f"   ❌ OEM has NO suppliers!")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    trace_oem_purchases()


