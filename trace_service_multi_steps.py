"""追踪Service多步运行"""
import sys
sys.path.append('.')

from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config
import numpy as np

config = load_config("config/config.json")
env = IndustryEnv(config.environment)
env.reset(options={"initial_firms": 50})

service = next((c for c in env.companies if sector_relations[c.sector_id].name == "Service"), None)
if not service:
    print("没找到Service")
    exit()

print("="*80)
print("Service多步追踪（10步）")
print("="*80)

print(f"\n初始资本: {service.capital:,.2f}")
print(f"收入率: {service.revenue_rate:.2f}")
print(f"产品单位成本: {service.product_unit_cost:.2f}")

action = np.zeros(env.action_space.shape, dtype=np.float32)

for step in range(1, 11):
    print(f"\n{'='*80}")
    print(f"STEP {step}")
    print(f"{'='*80}")
    
    before_capital = service.capital
    before_product_inv = service.product_inventory
    before_oem_inv = service.oem_inventory
    
    # 执行供应链
    for c in env.companies:
        c.reset_step_counters()
    
    env._simulate_supply_chain()
    
    if service not in env.companies:
        print("❌ Service已死亡!")
        break
    
    after_supply_chain_capital = service.capital
    purchase_spent = before_capital - after_supply_chain_capital
    
    # 记录财务
    revenue_amount = service.revenue
    cogs_amount = service.cogs_cost
    logistic_amount = service.logistic_cost
    
    # 计算step成本
    op_cost = service.op_cost_rate * after_supply_chain_capital
    capital_ratio = after_supply_chain_capital / max(env.max_capital, 1.0)
    mgmt_cost = after_supply_chain_capital * 0.001 * (capital_ratio ** 0.5)
    
    profit = revenue_amount - cogs_amount - logistic_amount - op_cost - mgmt_cost + service.fixed_income
    
    # 结算
    service.capital += profit
    service.revenue = 0.0
    service.cogs_cost = 0.0
    service.logistic_cost = 0.0
    
    final_capital = service.capital
    total_change = final_capital - before_capital
    
    # 显示信息
    print(f"资本: {before_capital:,.0f} → {final_capital:,.0f} ({total_change:+,.0f})")
    print(f"产品库存: {before_product_inv:.1f} → {service.product_inventory:.1f}")
    print(f"OEM库存: {before_oem_inv:.1f} → {service.oem_inventory:.1f}")
    
    print(f"\n活动:")
    print(f"  采购: {service.products_purchased_this_step:.1f} 个OEM")
    print(f"  生产: {service.products_produced_this_step:.1f} 个Service")
    print(f"  销售: {service.products_sold_this_step:.1f} 个Service")
    
    print(f"\n财务:")
    print(f"  收入: {revenue_amount:,.0f}")
    print(f"  COGS: {cogs_amount:,.0f}")
    print(f"  运营: {op_cost:,.0f}")
    print(f"  管理: {mgmt_cost:,.0f}")
    print(f"  运输: {logistic_amount:,.0f}")
    print(f"  营业利润: {profit:,.0f}")
    
    print(f"\n资本变化分解:")
    print(f"  采购支出: {-purchase_spent:,.0f}")
    print(f"  营业利润: {profit:+,.0f}")
    print(f"  总计: {total_change:+,.0f}")
    
    if total_change < 0:
        print(f"  ❌ 亏损原因:", end="")
        if revenue_amount == 0:
            print(f" 无收入（可能无库存销售）")
        elif purchase_spent > 0 and profit > 0:
            print(f" 采购占用资本（{purchase_spent:,.0f}）> 营业利润（{profit:,.0f}）")
        elif profit < 0:
            print(f" 营业亏损（成本{cogs_amount + op_cost + mgmt_cost:,.0f} > 收入{revenue_amount:,.0f}）")
    else:
        print(f"  ✅ 盈利!")

print(f"\n{'='*80}")
print(f"总结")
print(f"{'='*80}")
print(f"10步总变化: {service.capital - config.environment.initial_capital_min if hasattr(config.environment, 'initial_capital_min') else 'N/A'}")
print(f"最终资本: {service.capital:,.2f}")

