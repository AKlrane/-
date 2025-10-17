"""调试库存限制检查时机"""
import sys
sys.path.append('.')

import numpy as np
from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config

config = load_config("config/config.json")

print("="*80)
print("调试库存限制机制的执行")
print("="*80)

env = IndustryEnv(config.environment)
env.reset(options={"initial_firms": 30})

# 只运行一步，详细追踪
print(f"\n初始化: {len(env.companies)}家公司")

# 在step前记录每个公司的库存状态
pre_step_data = {}
for company in env.companies:
    if company.tier > 0:  # 只关注会购买的公司
        sector_name = sector_relations[company.sector_id].name
        inventory_value = company.product_inventory * company.revenue_rate
        inventory_ratio = inventory_value / max(company.capital, 1e-8)
        max_inventory_ratio = env.max_held_capital_rate.get(sector_name, 0.3)
        
        pre_step_data[id(company)] = {
            'sector': sector_name,
            'pre_inventory': company.product_inventory,
            'pre_inventory_value': inventory_value,
            'pre_inventory_ratio': inventory_ratio,
            'capital': company.capital,
            'max_ratio': max_inventory_ratio,
            'should_block': inventory_ratio > max_inventory_ratio
        }

print(f"\n{'='*80}")
print("Step前的状态（应该被阻止的公司）:")
print(f"{'='*80}")
for comp_id, data in pre_step_data.items():
    if data['should_block']:
        print(f"{data['sector']:15} 库存={data['pre_inventory']:>8.1f} "
              f"比例={data['pre_inventory_ratio']:>6.1%} (阈值={data['max_ratio']:.1%}) "
              f"→ 应该被阻止")

# 执行一步
action = np.zeros(env.action_space.shape)
obs, reward, terminated, truncated, info = env.step(action)

print(f"\n{'='*80}")
print("Step后的结果检查:")
print(f"{'='*80}")

for company in env.companies:
    comp_id = id(company)
    if comp_id not in pre_step_data:
        continue
    
    pre_data = pre_step_data[comp_id]
    did_purchase = company.products_purchased_this_step > 0
    
    if pre_data['should_block']:
        if did_purchase:
            print(f"❌ BUG! {pre_data['sector']:15} 应该被阻止但还是买了 {company.products_purchased_this_step:.1f} 单位")
            print(f"   购买前: 库存={pre_data['pre_inventory']:.1f}, 比例={pre_data['pre_inventory_ratio']:.1%} > {pre_data['max_ratio']:.1%}")
            print(f"   购买后: 库存={company.product_inventory:.1f}")
        else:
            print(f"✅ {pre_data['sector']:15} 正确阻止 (比例={pre_data['pre_inventory_ratio']:.1%} > {pre_data['max_ratio']:.1%})")
    elif did_purchase:
        print(f"✓  {pre_data['sector']:15} 正常购买 {company.products_purchased_this_step:.1f} 单位 "
              f"(比例={pre_data['pre_inventory_ratio']:.1%} < {pre_data['max_ratio']:.1%})")

print(f"\n{'='*80}")

