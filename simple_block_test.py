"""简单测试：修改env.py添加打印，直接观察阻止是否发生"""
import sys
sys.path.append('.')

import numpy as np
from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config

# 先读取配置
config = load_config("config/config.json")

print("="*80)
print("配置的max_held_capital_rate:")
for sector, rate in config.environment.max_held_capital_rate.items():
    print(f"  {sector:15} {rate:.1%}")
print("="*80)

# 临时修改env.py，添加打印
import env.env as env_module

original_simulate = env_module.IndustryEnv._simulate_supply_chain

def debug_simulate(self):
    """带调试输出的_simulate_supply_chain"""
    from env.sector import sector_relations
    from collections import defaultdict
    
    # 重置计数
    for company in self.companies:
        company.reset_step_counters()
    
    # Tier 0生产
    tier_0_companies = [c for c in self.companies if c.tier == 0]
    for company in tier_0_companies:
        company.produce_products()
    
    # 收集订单 - 带调试
    supplier_orders = defaultdict(list)
    K = 5
    
    blocked_count = 0
    allowed_count = 0
    
    for customer in self.companies:
        if customer.tier == 0 or not customer.suppliers:
            continue
        
        # 库存检查
        inventory_value = customer.product_inventory * customer.revenue_rate
        inventory_ratio = inventory_value / max(customer.capital, 1e-8)
        cust_sector_name = sector_relations[customer.sector_id].name
        max_inventory_ratio = self.max_held_capital_rate.get(cust_sector_name, 0.3)
        
        if inventory_ratio > max_inventory_ratio:
            blocked_count += 1
            if blocked_count <= 3:  # 只打印前3个
                print(f"  🚫 阻止 {cust_sector_name:15} 库存比例={inventory_ratio:.1%} > {max_inventory_ratio:.1%}")
            continue  # ← 这里应该阻止购买
        
        allowed_count += 1
        
        # 正常的购买逻辑...
        purchase_budget = customer.get_max_purchase_budget()
        if purchase_budget <= 0:
            continue
        
        cust_name = sector_relations[customer.sector_id].name
        if cust_name == "OEM":
            relevant_suppliers = [s for s in customer.suppliers 
                                 if sector_relations[s.sector_id].name in ("Parts", "Electronics", "Battery/Motor")]
            if relevant_suppliers:
                nearest_suppliers = sorted(relevant_suppliers, key=lambda s: customer.distance_to(s))[:K]
                budget_per_supplier = purchase_budget / len(nearest_suppliers)
                for supplier in nearest_suppliers:
                    unit_price = supplier.revenue_rate
                    units_requested = budget_per_supplier / max(unit_price, 1e-8)
                    if units_requested > 0:
                        dist = customer.distance_to(supplier)
                        supplier_orders[supplier].append((customer, dist, units_requested))
        else:
            nearest_suppliers = sorted(customer.suppliers, key=lambda s: customer.distance_to(s))[:K]
            if nearest_suppliers:
                budget_per_supplier = purchase_budget / len(nearest_suppliers)
                for supplier in nearest_suppliers:
                    unit_price = supplier.revenue_rate
                    units_requested = budget_per_supplier / max(unit_price, 1e-8)
                    if units_requested > 0:
                        dist = customer.distance_to(supplier)
                        supplier_orders[supplier].append((customer, dist, units_requested))
    
    print(f"  📊 统计: 阻止了{blocked_count}个, 允许了{allowed_count}个购买")
    
    # 继续原来的逻辑...
    original_simulate(self)

# 临时替换
env_module.IndustryEnv._simulate_supply_chain = debug_simulate

# 测试
env = IndustryEnv(config.environment)
env.reset(options={"initial_firms": 50})

print(f"\n运行3步测试:")
for i in range(3):
    print(f"\nStep {i+1}:")
    action = np.zeros(env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)

print("\n" + "="*80)
print("测试完成")

