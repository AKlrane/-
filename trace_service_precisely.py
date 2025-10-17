"""精确追踪Service的每一笔账"""
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
print("Service精确追踪")
print("="*80)

print(f"\n【初始状态】")
print(f"资本: {service.capital:,.2f}")
print(f"产品库存: {service.product_inventory:.2f}")
print(f"OEM库存: {service.oem_inventory:.2f}")
print(f"OEM单位成本: {service.input_cost_per_unit.get('oem', 0):.2f}")
print(f"产品单位成本: {service.product_unit_cost:.2f}")
print(f"收入率(单价): {service.revenue_rate:.2f}")
print(f"运营成本率: {service.op_cost_rate:.6f}")

# 找一个OEM供应商看价格
oem_suppliers = [c for c in service.suppliers if sector_relations[c.sector_id].name == "OEM"]
if oem_suppliers:
    print(f"\nOEM供应商单价示例:")
    for i, oem in enumerate(oem_suppliers[:3]):
        print(f"  {i+1}. OEM单价={oem.revenue_rate:.2f}, 库存={oem.product_inventory:.2f}")

before_capital = service.capital

# 执行step
env._simulate_supply_chain()

after_supply_chain_capital = service.capital
purchase_spent = before_capital - after_supply_chain_capital

print(f"\n【供应链模拟后】")
print(f"资本: {before_capital:,.2f} → {after_supply_chain_capital:,.2f}")
print(f"采购支出: {purchase_spent:,.2f}")
print(f"采购数量: {service.products_purchased_this_step:.2f}")
if service.products_purchased_this_step > 0:
    avg_price = purchase_spent / service.products_purchased_this_step
    print(f"平均采购单价: {avg_price:.2f}")

print(f"\n产品库存: {service.product_inventory:.2f}")
print(f"OEM库存: {service.oem_inventory:.2f}")
print(f"生产: {service.products_produced_this_step:.2f}")
print(f"销售: {service.products_sold_this_step:.2f}")

print(f"\n财务累计（step()前）:")
print(f"  Revenue: {service.revenue:,.2f}")
print(f"  COGS: {service.cogs_cost:,.2f}")
print(f"  Logistic: {service.logistic_cost:,.2f}")

# 计算step成本
op_cost = service.op_cost_rate * after_supply_chain_capital
capital_ratio = after_supply_chain_capital / max(env.max_capital, 1.0)
mgmt_cost = after_supply_chain_capital * 0.001 * (capital_ratio ** 0.5)

print(f"\nStep结算计算:")
print(f"  运营成本: {after_supply_chain_capital:,.2f} × {service.op_cost_rate:.6f} = {op_cost:,.2f}")
print(f"  管理成本: {mgmt_cost:,.2f}")
print(f"  Fixed income: {service.fixed_income:.2f}")

profit = service.revenue - service.cogs_cost - service.logistic_cost - op_cost - mgmt_cost + service.fixed_income

print(f"\n利润 = {service.revenue:,.2f} - {service.cogs_cost:,.2f} - {service.logistic_cost:,.2f} - {op_cost:,.2f} - {mgmt_cost:,.2f} + {service.fixed_income:.2f}")
print(f"     = {profit:,.2f}")

# 执行结算
service.capital += profit
service.revenue = 0.0
service.cogs_cost = 0.0
service.logistic_cost = 0.0

final_capital = service.capital
total_change = final_capital - before_capital

print(f"\n【最终】")
print(f"期初资本: {before_capital:,.2f}")
print(f"期末资本: {final_capital:,.2f}")
print(f"总变化: {total_change:,.2f}")

print(f"\n分解:")
print(f"  采购支出: -{purchase_spent:,.2f}")
print(f"  营业利润: +{profit:,.2f}")
print(f"  合计: {-purchase_spent + profit:,.2f}")

if abs(total_change - (-purchase_spent + profit)) < 0.01:
    print(f"\n✅ 账目完全吻合！")
else:
    print(f"\n❌ 有差异: {total_change - (-purchase_spent + profit):.2f}")

# 分析为什么亏
print(f"\n{'='*80}")
print(f"亏损分析")
print(f"{'='*80}")

if total_change < 0:
    毛利 = service.products_sold_this_step * service.revenue_rate - service.products_sold_this_step * service.product_unit_cost
    print(f"\n毛利分析:")
    print(f"  销售收入: {service.products_sold_this_step:.2f} × {service.revenue_rate:.2f} = {service.products_sold_this_step * service.revenue_rate:,.2f}")
    print(f"  COGS: {service.products_sold_this_step:.2f} × {service.product_unit_cost:.2f} = {service.products_sold_this_step * service.product_unit_cost:,.2f}")
    print(f"  毛利: {毛利:,.2f}")
    if service.products_sold_this_step > 0:
        print(f"  毛利率: {毛利 / (service.products_sold_this_step * service.revenue_rate) * 100:.1f}%")
    else:
        print(f"  ⚠️  本期没有销售")
    
    if profit > 0:
        print(f"\n✅ 营业是盈利的（{profit:,.2f}）")
        print(f"❌ 但采购支出过大（{purchase_spent:,.2f}），资本被占用")
        print(f"   采购/销售比: {service.products_purchased_this_step / max(service.products_sold_this_step, 1):.2f}倍")
    else:
        print(f"\n❌ 营业本身亏损（{profit:,.2f}）")
        if service.cogs_cost / max(service.revenue, 1) > 0.8:
            print(f"   主因：COGS占收入{service.cogs_cost / service.revenue * 100:.1f}%（过高）")
        if op_cost + mgmt_cost > 毛利:
            print(f"   主因：管理和运营成本{op_cost + mgmt_cost:,.2f} > 毛利{毛利:,.2f}")

