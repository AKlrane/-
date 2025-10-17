"""Debug script to trace Service company financials step by step."""

import numpy as np
from config import Config
from env import IndustryEnv
from env.sector import sector_relations


def debug_service_financials():
    config = Config()
    config.environment.logistic_cost_rate = 0.03
    config.environment.size = 40.0
    
    env = IndustryEnv(config.environment)
    obs, _ = env.reset(options={"initial_firms": 50})
    
    # Find a Service company
    service_companies = [c for c in env.companies if sector_relations[c.sector_id].name == "Service"]
    
    if not service_companies:
        print("No Service companies found!")
        return
    
    service = service_companies[0]
    print(f"\n=== Tracking Service Company at {service.location} ===")
    print(f"Initial capital: {service.capital:,.2f}")
    print(f"Revenue rate (selling price): {service.revenue_rate}")
    print(f"Unit COGS: {service.unit_cogs}")
    print(f"Product unit cost: {service.product_unit_cost}")
    
    # Run 10 steps and track financials
    for step in range(1, 11):
        print(f"\n--- Step {step} ---")
        
        # Before step
        print(f"Before step:")
        capital_before = service.capital
        print(f"  Capital: {capital_before:,.2f}")
        print(f"  Product inventory: {service.product_inventory:.2f}")
        print(f"  OEM inventory: {service.oem_inventory:.2f}")
        print(f"  Input cost per unit (OEM): {service.input_cost_per_unit.get('oem', 0.0):.2f}")
        print(f"  Product unit cost: {service.product_unit_cost:.2f}")
        
        # Do step
        zero_action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Check if this is an investment action targeting Service
        if zero_action[0] < 0.5:  # Invest action
            print(f"  Action type: INVEST with amount_normalized={zero_action[1]:.3f}")
        
        obs, reward, terminated, truncated, info = env.step(zero_action)
        
        capital_after = service.capital
        capital_change = capital_after - capital_before
        
        # After step
        print(f"After step:")
        print(f"  Capital: {capital_after:,.2f} (change: {capital_change:+,.2f})")
        print(f"  Revenue: {service.revenue:.2f}")
        print(f"  COGS cost: {service.cogs_cost:.2f}")
        print(f"  Logistic cost: {service.logistic_cost:.2f}")
        print(f"  Products produced: {service.products_produced_this_step:.2f}")
        print(f"  Products purchased: {service.products_purchased_this_step:.2f}")
        print(f"  Products sold: {service.products_sold_this_step:.2f}")
        print(f"  Product inventory: {service.product_inventory:.2f}")
        print(f"  OEM inventory: {service.oem_inventory:.2f}")
        
        # Calculate expected profit
        op_cost = 0.05 * service.capital
        max_capital = 100000000.0
        capital_ratio = max(service.capital, 0.0) / max_capital
        management_cost = max(service.capital, 0.0) * 0.001 * (capital_ratio ** 0.5)
        fixed_income = -20.0
        
        print(f"  Operating cost: {op_cost:.2f}")
        print(f"  Management cost: {management_cost:.2f}")
        print(f"  Fixed income: {fixed_income:.2f}")
        print(f"  Expected profit: {service.revenue - service.cogs_cost - service.logistic_cost - op_cost - management_cost + fixed_income:.2f}")
        
        if terminated or truncated:
            print(f"\nSimulation ended at step {step}")
            break


if __name__ == "__main__":
    debug_service_financials()

