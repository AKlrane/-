"""追踪Service公司为什么会死亡"""
import sys
sys.path.append('.')

import numpy as np
from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config

config = load_config("config/config.json")

print("="*80)
print("追踪Service公司财务状况")
print("="*80)

env = IndustryEnv(config.environment)
env.reset(options={"initial_firms": 50})

# 找几个Service公司追踪
service_companies = [c for c in env.companies if sector_relations[c.sector_id].name == "Service"]
if not service_companies:
    print("没有Service公司！")
    sys.exit(1)

tracked = service_companies[:3]  # 追踪前3个

print(f"\n追踪 {len(tracked)} 家Service公司")
print(f"配置:")
print(f"  Service售价: {config.environment.tier_prices.get('Service', 0)}")
print(f"  OEM售价: {config.environment.tier_prices.get('OEM', 0)}")
print(f"  Service生产: 2 OEM -> 1 Service")
print(f"  理论成本: 2 × {config.environment.tier_prices.get('OEM', 0)} = {2 * config.environment.tier_prices.get('OEM', 0)}")
print(f"  理论毛利: {config.environment.tier_prices.get('Service', 0)} - {2 * config.environment.tier_prices.get('OEM', 0)} = {config.environment.tier_prices.get('Service', 0) - 2 * config.environment.tier_prices.get('OEM', 0)}")
print(f"  tier_production_ratios[Service]: {config.environment.tier_production_ratios.get('Service', 0)}")
print(f"  max_held_capital_rate[Service]: {config.environment.max_held_capital_rate.get('Service', 0)}")

for step in range(50):
    action = np.zeros(env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if step % 10 == 0:
        print(f"\n{'='*80}")
        print(f"Step {step + 1}:")
        print(f"{'='*80}")
        
        for i, service in enumerate(tracked):
            if service not in env.companies:
                print(f"\n❌ Service #{i+1} 已死亡")
                continue
            
            # 计算各项指标
            inventory_value = service.product_inventory * service.revenue_rate
            inventory_ratio = inventory_value / max(service.capital, 1e-8)
            max_inventory_ratio = env.max_held_capital_rate.get("Service", 0.3)
            purchase_budget = service.get_max_purchase_budget()
            
            print(f"\n📊 Service #{i+1}:")
            print(f"  Capital: {service.capital:>10,.0f}")
            print(f"  Revenue (step): {service.revenue:>10,.0f}")
            print(f"  Product库存: {service.product_inventory:>8.1f} (价值={inventory_value:>10,.0f}, 比例={inventory_ratio:.1%})")
            print(f"  OEM库存: {service.oem_inventory:>8.1f}")
            
            print(f"  购买预算: {purchase_budget:>10,.0f} (capital × {config.environment.tier_production_ratios.get('Service', 0)})")
            print(f"  库存限制: {inventory_ratio:.1%} {'>' if inventory_ratio > max_inventory_ratio else '<='} {max_inventory_ratio:.1%} → {'🚫阻止购买' if inventory_ratio > max_inventory_ratio else '✅可购买'}")
            
            print(f"  本步统计:")
            print(f"    购买: {service.products_purchased_this_step:>8.1f}")
            print(f"    生产: {service.products_produced_this_step:>8.1f}")
            print(f"    销售: {service.products_sold_this_step:>8.1f}")
            
            if service.products_sold_this_step > 0:
                avg_unit_cost = service.product_unit_cost
                revenue_per_unit = service.revenue_rate
                gross_margin = revenue_per_unit - avg_unit_cost
                print(f"  单位经济:")
                print(f"    售价: {revenue_per_unit:>8.1f}")
                print(f"    COGS: {avg_unit_cost:>8.1f}")
                print(f"    毛利: {gross_margin:>8.1f} ({gross_margin/revenue_per_unit*100:.1f}%)")
            
            # 找到上游OEM供应商
            if service.suppliers:
                print(f"  上游OEM供应商: {len(service.suppliers)}家")
                nearest_oem = sorted(service.suppliers, key=lambda s: service.distance_to(s))[:3]
                for j, oem in enumerate(nearest_oem):
                    print(f"    OEM#{j+1}: capital={oem.capital:>10,.0f}, 库存={oem.product_inventory:>6.1f}, 距离={service.distance_to(oem):.2f}")

print(f"\n{'='*80}")
print("追踪完成")
print(f"{'='*80}")

