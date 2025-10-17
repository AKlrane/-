"""
Debug script to analyze why Parts/Electronics/Battery companies are dying.
Track detailed financials and supply chain for these sectors.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.env import IndustryEnv
from config.config import Config
from env.sector import sector_relations
import numpy as np


def analyze_peb_survival():
    """Analyze P/E/B sector survival issues."""
    print("=" * 80)
    print("P/E/B SECTOR SURVIVAL ANALYSIS")
    print("=" * 80)
    
    config = Config.from_json("config/config.json")
    
    # Print key config parameters
    print("\n📊 Key Configuration:")
    print(f"   Tier Prices: {config.environment.tier_prices}")
    print(f"   Tier COGS: {config.environment.tier_cogs}")
    print(f"   Op Cost Rate: {config.environment.op_cost_rate}")
    print(f"   Logistic Cost Rate: {config.environment.logistic_cost_rate}")
    print(f"   Production Capacity Ratio: {config.environment.production_capacity_ratio}")
    print(f"   Purchase Budget Ratio: {config.environment.purchase_budget_ratio}")
    
    # Create environment
    env = IndustryEnv(config.environment)
    obs, info = env.reset(options={"initial_firms": 30})
    
    print(f"\n🏭 Initial State:")
    print(f"   Total companies: {len(env.companies)}")
    
    # Track P/E/B companies
    def get_sector_companies(sector_name):
        return [c for c in env.companies if sector_relations[c.sector_id].name == sector_name]
    
    # Run simulation for several steps and track P/E/B
    num_steps = 10
    
    for step in range(1, num_steps + 1):
        # Take zero action
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(zero_action)
        
        if step % 2 == 0 or step == 1:
            print(f"\n{'=' * 80}")
            print(f"STEP {step}")
            print('=' * 80)
            
            # Analyze each P/E/B sector
            for sector_name in ["Parts", "Electronics", "Battery/Motor"]:
                companies = get_sector_companies(sector_name)
                
                if not companies:
                    print(f"\n❌ {sector_name}: ALL DEAD")
                    continue
                
                print(f"\n📦 {sector_name} Sector ({len(companies)} companies alive):")
                
                # Detailed analysis of each company
                for i, company in enumerate(companies[:3]):  # Show top 3
                    sup_name = sector_relations[company.sector_id].name
                    
                    print(f"\n   Company #{i+1}:")
                    print(f"      Capital: ${company.capital:,.0f}")
                    print(f"      Revenue Rate (Price): ${company.revenue_rate:.2f}")
                    
                    # Calculate costs
                    op_cost = company.op_cost_rate * company.capital
                    capital_ratio = max(company.capital, 0.0) / max(config.environment.max_capital, 1.0)
                    management_cost = max(company.capital, 0.0) * 0.001 * (capital_ratio ** 0.5)
                    
                    print(f"      Op Cost Rate: {company.op_cost_rate:.4f}")
                    print(f"      Op Cost: ${op_cost:,.2f}")
                    print(f"      Management Cost: ${management_cost:,.2f}")
                    print(f"      Fixed Income: ${company.fixed_income:.2f}")
                    
                    # Production info
                    raw_inv = company.raw_inventory
                    prod_inv = company.product_inventory
                    max_prod = company.get_max_production()
                    max_purchase = company.get_max_purchase_budget()
                    
                    print(f"      Raw Inventory: {raw_inv:.2f}")
                    print(f"      Product Inventory: {prod_inv:.2f}")
                    print(f"      Max Production: {max_prod:.2f}")
                    print(f"      Max Purchase Budget: ${max_purchase:,.2f}")
                    print(f"      Products Produced: {company.products_produced_this_step:.2f}")
                    print(f"      Products Sold: {company.products_sold_this_step:.2f}")
                    
                    # Revenue and cost breakdown
                    print(f"      Last Revenue: ${company.revenue:,.2f}")
                    print(f"      Last COGS: ${company.cogs_cost:,.2f}")
                    print(f"      Last Logistic Cost: ${company.logistic_cost:,.2f}")
                    
                    # Calculate profit
                    total_cost = op_cost + management_cost + company.logistic_cost + company.cogs_cost
                    profit = company.revenue - total_cost + company.fixed_income
                    
                    print(f"      Estimated Profit: ${profit:,.2f}")
                    
                    # Supplier/customer info
                    print(f"      Suppliers: {len(company.suppliers)}")
                    print(f"      Customers: {len(company.customers)}")
                    
                    # Check profitability
                    if profit < 0:
                        print(f"      ⚠️  LOSING MONEY!")
                        
                        # Identify problem
                        if company.products_sold_this_step == 0:
                            print(f"      ❌ Problem: NO SALES (no customers buying)")
                        elif company.products_produced_this_step == 0:
                            print(f"      ❌ Problem: NO PRODUCTION (no raw materials)")
                        elif company.revenue < total_cost:
                            print(f"      ❌ Problem: REVENUE < COSTS")
                            print(f"         Revenue: ${company.revenue:,.2f}")
                            print(f"         Total Cost: ${total_cost:,.2f}")
                            print(f"         Gap: ${total_cost - company.revenue:,.2f}")
                    
                # Sector summary
                avg_capital = np.mean([c.capital for c in companies])
                total_revenue = sum([c.revenue for c in companies])
                total_production = sum([c.products_produced_this_step for c in companies])
                total_sales = sum([c.products_sold_this_step for c in companies])
                
                print(f"\n   {sector_name} Sector Summary:")
                print(f"      Avg Capital: ${avg_capital:,.0f}")
                print(f"      Total Revenue: ${total_revenue:,.2f}")
                print(f"      Total Production: {total_production:.2f}")
                print(f"      Total Sales: {total_sales:.2f}")
    
    # Final analysis
    print("\n" + "=" * 80)
    print("FINAL DIAGNOSIS")
    print("=" * 80)
    
    parts = get_sector_companies("Parts")
    elec = get_sector_companies("Electronics")
    batt = get_sector_companies("Battery/Motor")
    oem = get_sector_companies("OEM")
    raw = get_sector_companies("Raw")
    
    print(f"\n📊 Survival Status:")
    print(f"   Raw: {len(raw)} alive")
    print(f"   Parts: {len(parts)} alive")
    print(f"   Electronics: {len(elec)} alive")
    print(f"   Battery/Motor: {len(batt)} alive")
    print(f"   OEM: {len(oem)} alive")
    
    # Analyze pricing vs costs
    print(f"\n💰 Price vs Cost Analysis:")
    
    raw_price = config.environment.tier_prices.get("Raw", 1.0)
    parts_price = config.environment.tier_prices.get("Parts", 5.0)
    elec_price = config.environment.tier_prices.get("Electronics", 12.0)
    batt_price = config.environment.tier_prices.get("Battery/Motor", 35.0)
    
    print(f"\n   Parts: 3 raw → 1 parts")
    print(f"      Input cost: 3 × ${raw_price:.2f} = ${3 * raw_price:.2f}")
    print(f"      Output price: ${parts_price:.2f}")
    print(f"      Gross margin: ${parts_price - 3 * raw_price:.2f}")
    print(f"      Margin %: {((parts_price - 3 * raw_price) / parts_price * 100):.1f}%")
    
    print(f"\n   Electronics: 7 raw → 1 electronics")
    print(f"      Input cost: 7 × ${raw_price:.2f} = ${7 * raw_price:.2f}")
    print(f"      Output price: ${elec_price:.2f}")
    print(f"      Gross margin: ${elec_price - 7 * raw_price:.2f}")
    print(f"      Margin %: {((elec_price - 7 * raw_price) / elec_price * 100):.1f}%")
    
    print(f"\n   Battery/Motor: 20 raw → 1 battery")
    print(f"      Input cost: 20 × ${raw_price:.2f} = ${20 * raw_price:.2f}")
    print(f"      Output price: ${batt_price:.2f}")
    print(f"      Gross margin: ${batt_price - 20 * raw_price:.2f}")
    print(f"      Margin %: {((batt_price - 20 * raw_price) / batt_price * 100):.1f}%")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_peb_survival()


