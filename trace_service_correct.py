"""正确追踪Service的会计逻辑（修复后）"""
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
print("Service会计追踪（正确版本）")
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
    
    # 执行供应链（包含采购、生产、销售）
    for c in env.companies:
        c.reset_step_counters()
    
    env._simulate_supply_chain()
    
    if service not in env.companies:
        print("❌ Service已死亡!")
        break
    
    # 供应链后，capital只会因为运输成本减少
    after_supply_chain_capital = service.capital
    logistics_paid = before_capital - after_supply_chain_capital
    
    # 记录累计的收入和成本（还未结算）
    revenue_amount = service.revenue
    cogs_amount = service.cogs_cost
    
    # 计算step结算的成本
    op_cost = service.op_cost_rate * after_supply_chain_capital
    capital_ratio = after_supply_chain_capital / max(env.max_capital, 1.0)
    mgmt_cost = after_supply_chain_capital * 0.001 * (capital_ratio ** 0.5)
    
    # 正确的profit计算（不包括logistics，已经扣过了）
    profit = revenue_amount - cogs_amount - op_cost - mgmt_cost + service.fixed_income
    
    # 手动执行结算
    service.capital += profit
    service.revenue = 0.0
    service.cogs_cost = 0.0
    service.logistic_cost = 0.0
    
    final_capital = service.capital
    total_change = final_capital - before_capital
    
    # 显示
    print(f"资本: {before_capital:,.0f} → {final_capital:,.0f} ({total_change:+,.0f})")
    print(f"产品库存: {service.product_inventory:.1f}")
    print(f"OEM库存: {service.oem_inventory:.1f}")
    
    print(f"\n本步活动:")
    print(f"  采购: {service.products_purchased_this_step:.1f} 个OEM")
    print(f"  生产: {service.products_produced_this_step:.1f} 个Service")
    print(f"  销售: {service.products_sold_this_step:.1f} 个Service")
    
    print(f"\n财务明细:")
    print(f"  1. 运输成本(立即支付): {logistics_paid:,.0f}")
    print(f"  2. 收入(记账): {revenue_amount:,.0f}")
    print(f"  3. COGS(记账): {cogs_amount:,.0f}")
    print(f"  4. 运营成本: {op_cost:,.0f}")
    print(f"  5. 管理成本: {mgmt_cost:,.0f}")
    print(f"  6. Fixed income: {service.fixed_income:.0f}")
    
    print(f"\n资本变化计算:")
    print(f"  期初: {before_capital:,.0f}")
    print(f"  - 运输(现金): {logistics_paid:,.0f}")
    print(f"  = 供应链后: {after_supply_chain_capital:,.0f}")
    print(f"  + 营业利润: {profit:+,.0f}")
    print(f"    (= {revenue_amount:,.0f}收入 - {cogs_amount:,.0f}COGS - {op_cost:,.0f}运营 - {mgmt_cost:,.0f}管理 + {service.fixed_income:.0f}固定)")
    print(f"  = 期末: {final_capital:,.0f}")
    
    # 验证
    calculated_final = after_supply_chain_capital + profit
    if abs(calculated_final - final_capital) < 0.01:
        print(f"  ✅ 会计平衡")
    else:
        print(f"  ❌ 不平衡！差异={calculated_final - final_capital:.2f}")
    
    # 分析
    if total_change < 0:
        print(f"\n  亏损 {-total_change:,.0f}:")
        if revenue_amount == 0:
            print(f"    - 无收入（没有库存销售）")
        else:
            gross_profit = revenue_amount - cogs_amount
            if gross_profit < 0:
                print(f"    - 毛利为负（{gross_profit:,.0f}）")
            elif gross_profit < op_cost + mgmt_cost:
                print(f"    - 毛利（{gross_profit:,.0f}）< 运营管理成本（{op_cost + mgmt_cost:,.0f}）")
        if logistics_paid > 0:
            print(f"    - 支付运输费 {logistics_paid:,.0f}")
    else:
        print(f"\n  ✅ 盈利 {total_change:,.0f}")

print(f"\n{'='*80}")
print(f"10步总结")
print(f"{'='*80}")
print(f"最终资本: {service.capital:,.2f}")

