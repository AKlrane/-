"""详细诊断Service为什么死得快"""
import sys
sys.path.append('.')

import numpy as np
from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config

config = load_config("config/config.json")

print("="*80)
print("诊断：为什么Service死得最快？")
print("="*80)

print("\n【配置参数】")
print(f"  Service售价: {config.environment.tier_prices.get('Service', 0)}")
print(f"  OEM售价: {config.environment.tier_prices.get('OEM', 0)}")
print(f"  固定成本/step: {config.environment.fixed_cost_per_step}")
print(f"  运输成本率: {config.environment.logistic_cost_rate}")
print(f"  Service购买能力: {config.environment.tier_production_ratios.get('Service', 0)} × capital")
print(f"  Service库存限制: {config.environment.max_held_capital_rate.get('Service', 0)} × capital")

print("\n【理论利润分析】")
oem_price = config.environment.tier_prices.get('OEM', 0)
service_price = config.environment.tier_prices.get('Service', 0)
print(f"  生产配方: 2 OEM -> 1 Service")
print(f"  COGS成本: 2 × {oem_price} = {2 * oem_price}")
print(f"  销售收入: {service_price}")
print(f"  理论毛利: {service_price - 2 * oem_price}")
print(f"  毛利率: {(service_price - 2 * oem_price) / service_price * 100:.1f}%")

env = IndustryEnv(config.environment)
env.reset(options={"initial_firms": 50})

service_companies = [c for c in env.companies if sector_relations[c.sector_id].name == "Service"]
print(f"\n【初始状态】")
print(f"  Service公司数量: {len(service_companies)}")
print(f"  初始capital范围: {min(c.capital for c in service_companies):,.0f} - {max(c.capital for c in service_companies):,.0f}")

# 追踪所有Service公司
service_ids = {id(c): i for i, c in enumerate(service_companies)}
survival_data = []

for step in range(100):
    action = np.zeros(env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 收集存活的Service公司数据
    alive_services = [c for c in env.companies if sector_relations[c.sector_id].name == "Service"]
    
    if step % 20 == 19:  # 每20步分析一次
        print(f"\n{'='*80}")
        print(f"Step {step + 1}:")
        print(f"  存活Service: {len(alive_services)}/{len(service_companies)}")
        
        if alive_services:
            # 统计数据
            total_purchased = sum(c.products_purchased_this_step for c in alive_services)
            total_produced = sum(c.products_produced_this_step for c in alive_services)
            total_sold = sum(c.products_sold_this_step for c in alive_services)
            
            print(f"\n  平均数据（每家Service）:")
            print(f"    购买: {total_purchased / len(alive_services):.1f}")
            print(f"    生产: {total_produced / len(alive_services):.1f}")
            print(f"    销售: {total_sold / len(alive_services):.1f}")
            print(f"    Capital: {sum(c.capital for c in alive_services) / len(alive_services):,.0f}")
            
            # 找出购买=0的公司
            no_purchase = [c for c in alive_services if c.products_purchased_this_step == 0]
            if no_purchase:
                print(f"\n  ⚠️  {len(no_purchase)}/{len(alive_services)} 家Service本步买不到货！")
                
                # 分析为什么买不到货
                sample = no_purchase[0]
                inventory_value = sample.product_inventory * sample.revenue_rate
                inventory_ratio = inventory_value / max(sample.capital, 1e-8)
                max_inventory_ratio = env.max_held_capital_rate.get("Service", 0.3)
                
                if inventory_ratio > max_inventory_ratio:
                    print(f"      原因: 库存阻止 ({inventory_ratio:.1%} > {max_inventory_ratio:.1%})")
                elif sample.capital < 1000:
                    print(f"      原因: capital太低 ({sample.capital:,.0f})")
                elif not sample.suppliers:
                    print(f"      原因: 没有上游供应商")
                else:
                    # 检查上游OEM库存
                    oem_total_inventory = sum(s.product_inventory for s in sample.suppliers)
                    print(f"      上游OEM总库存: {oem_total_inventory:.0f}")
                    if oem_total_inventory < 1:
                        print(f"      原因: 上游OEM没有库存！")
                    else:
                        print(f"      原因: 被其他Service抢走了（距离/优先级问题）")
            
            # 分析死亡的公司
            dead_services = [c_id for c_id in service_ids if not any(id(c) == c_id for c in alive_services)]
            if len(dead_services) > len(survival_data):
                newly_dead = len(dead_services) - len(survival_data)
                print(f"\n  💀 本周期新死亡: {newly_dead}家")
        
        survival_data.append(len(alive_services))

print(f"\n{'='*80}")
print("【结论分析】")
print(f"{'='*80}")

print(f"\n存活率变化:")
for i, count in enumerate(survival_data):
    step = (i + 1) * 20
    rate = count / len(service_companies) * 100
    print(f"  Step {step:3d}: {count:2d}/{len(service_companies)} ({rate:5.1f}%)")

print(f"\n可能的死因（按重要性排序）:")
print(f"  1. 🚫 买不到货：上游OEM供应不足，远距离Service抢不到")
print(f"  2. 💸 运输成本：logistic_cost_rate={config.environment.logistic_cost_rate}，距离远成本高")
print(f"  3. 📉 固定成本：每步扣{config.environment.fixed_cost_per_step}，买不到货时持续亏损")
print(f"  4. 🏦 库存阻止：库存>{config.environment.max_held_capital_rate.get('Service', 0):.0%} capital时不能购买")

print(f"\n建议解决方案:")
print(f"  1. 提高OEM产能: tier_production_ratios['OEM'] 从 {config.environment.tier_production_ratios.get('OEM')} 提高到 0.25+")
print(f"  2. 降低运输成本: logistic_cost_rate 从 {config.environment.logistic_cost_rate} 降低到 0.05")
print(f"  3. 提高Service价格或降低OEM价格以增加利润margin")
print(f"  4. 增加初始OEM公司数量")

print(f"\n{'='*80}")

