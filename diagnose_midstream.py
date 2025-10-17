"""诊断中游公司（Parts/Electronics/Battery）为什么崩溃"""
import sys
sys.path.append('.')

from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config
import numpy as np

config = load_config("config/config.json")
env = IndustryEnv(config.environment)
env.reset(options={"initial_firms": 50})

print("="*80)
print("中游公司（P/E/B）崩溃诊断")
print("="*80)

# 找一个Parts公司追踪
parts = next((c for c in env.companies if sector_relations[c.sector_id].name == "Parts"), None)
if not parts:
    print("没找到Parts")
    exit()

print(f"\n【Parts公司初始状态】")
print(f"资本: {parts.capital:,.0f}")
print(f"采购比率: {parts.production_capacity_ratio}")
print(f"单价: {parts.revenue_rate:.2f}")
print(f"运营成本倍率: {sector_relations[parts.sector_id].operating_cost_multiplier}")
print(f"产品单位成本: {parts.product_unit_cost:.2f}")

# 检查转换比率
print(f"\n【转换系数】")
print(f"Parts: 3 Raw → 1 Parts")
print(f"理论上，1个Raw成本={config.environment.tier_prices.get('Raw', 1.0):.2f}")
print(f"所以1个Parts成本=3×{config.environment.tier_prices.get('Raw', 1.0):.2f}={3*config.environment.tier_prices.get('Raw', 1.0):.2f}")
print(f"Parts售价={parts.revenue_rate:.2f}")
毛利 = parts.revenue_rate - 3*config.environment.tier_prices.get('Raw', 1.0)
毛利率 = 毛利 / parts.revenue_rate * 100
print(f"毛利={毛利:.2f}, 毛利率={毛利率:.1f}%")
if 毛利率 < 20:
    print(f"⚠️  毛利率太低！")

action = np.zeros(env.action_space.shape, dtype=np.float32)

print(f"\n{'='*80}")
print(f"运行5步观察")
print(f"{'='*80}")

for step in range(1, 6):
    print(f"\n--- STEP {step} ---")
    
    before_capital = parts.capital
    before_raw = parts.raw_inventory
    before_product = parts.product_inventory
    
    # 执行
    for c in env.companies:
        c.reset_step_counters()
    env._simulate_supply_chain()
    
    if parts not in env.companies:
        print("❌ Parts已死亡!")
        break
    
    after_supply_capital = parts.capital
    logistics_paid = before_capital - after_supply_capital
    
    revenue = parts.revenue
    cogs = parts.cogs_cost
    
    op_cost = parts.op_cost_rate * after_supply_capital
    capital_ratio = after_supply_capital / max(env.max_capital, 1.0)
    mgmt_cost = after_supply_capital * 0.001 * (capital_ratio ** 0.5)
    
    profit = revenue - cogs - op_cost - mgmt_cost + parts.fixed_income
    
    parts.capital += profit
    parts.revenue = 0.0
    parts.cogs_cost = 0.0
    parts.logistic_cost = 0.0
    
    final_capital = parts.capital
    
    print(f"资本: {before_capital:,.0f} → {final_capital:,.0f} ({final_capital - before_capital:+,.0f})")
    print(f"Raw库存: {before_raw:.1f} → {parts.raw_inventory:.1f}")
    print(f"产品库存: {before_product:.1f} → {parts.product_inventory:.1f}")
    print(f"采购: {parts.products_purchased_this_step:.1f}, 生产: {parts.products_produced_this_step:.1f}, 销售: {parts.products_sold_this_step:.1f}")
    
    if parts.products_sold_this_step > 0:
        单位收入 = revenue / parts.products_sold_this_step
        单位成本 = cogs / parts.products_sold_this_step
        单位毛利 = 单位收入 - 单位成本
        print(f"单位经济: 收入={单位收入:.2f}, 成本={单位成本:.2f}, 毛利={单位毛利:.2f}")
    
    print(f"收入: {revenue:,.0f}, COGS: {cogs:,.0f}, 运营: {op_cost:.0f}, 管理: {mgmt_cost:.0f}, 运输: {logistics_paid:.0f}")
    print(f"营业利润: {profit:+,.0f}")
    
    # 分析问题
    if final_capital < before_capital:
        if revenue == 0:
            print(f"❌ 问题：无收入（库存={parts.product_inventory:.0f}，没卖出去）")
        elif cogs > revenue:
            print(f"❌ 问题：COGS({cogs:,.0f}) > 收入({revenue:,.0f})，成本倒挂！")
        elif op_cost + mgmt_cost > revenue - cogs:
            print(f"❌ 问题：运营管理成本({op_cost + mgmt_cost:.0f}) > 毛利({revenue - cogs:.0f})")
    
    # 检查供需
    if parts.products_produced_this_step > parts.products_sold_this_step * 2:
        print(f"⚠️  生产({parts.products_produced_this_step:.0f}) >> 销售({parts.products_sold_this_step:.0f})，产品积压")
    if parts.products_purchased_this_step == 0:
        print(f"⚠️  没有采购Raw，可能Raw供应不足")

print(f"\n{'='*80}")
print(f"诊断结论")
print(f"{'='*80}")

# 检查价格链
raw_price = config.environment.tier_prices.get('Raw', 1.0)
parts_price = config.environment.tier_prices.get('Parts', 6.0)
elec_price = config.environment.tier_prices.get('Electronics', 15.0)
batt_price = config.environment.tier_prices.get('Battery/Motor', 40.0)
oem_price = config.environment.tier_prices.get('OEM', 200.0)

print(f"\n价格链检查:")
print(f"  Raw: {raw_price:.2f}")
print(f"  Parts: {parts_price:.2f} (成本={3*raw_price:.2f}, 毛利={parts_price - 3*raw_price:.2f})")
print(f"  Electronics: {elec_price:.2f} (成本={7*raw_price:.2f}, 毛利={elec_price - 7*raw_price:.2f})")
print(f"  Battery: {batt_price:.2f} (成本={20*raw_price:.2f}, 毛利={batt_price - 20*raw_price:.2f})")
print(f"  OEM: {oem_price:.2f}")

if parts_price <= 3*raw_price:
    print(f"\n❌ Parts价格过低！售价{parts_price}不足以覆盖成本{3*raw_price}")
    print(f"   建议：Parts价格至少应为 {3*raw_price*1.5:.2f}")

if elec_price <= 7*raw_price:
    print(f"\n❌ Electronics价格过低！售价{elec_price}不足以覆盖成本{7*raw_price}")
    print(f"   建议：Electronics价格至少应为 {7*raw_price*1.5:.2f}")

if batt_price <= 20*raw_price:
    print(f"\n❌ Battery价格过低！售价{batt_price}不足以覆盖成本{20*raw_price}")
    print(f"   建议：Battery价格至少应为 {20*raw_price*1.5:.2f}")

print(f"\n运营成本检查:")
sample_capital = 10000
op_cost_sample = sample_capital * parts.op_cost_rate
print(f"  Parts (capital={sample_capital:,.0f}):")
print(f"    运营成本 = {sample_capital} × {parts.op_cost_rate:.6f} = {op_cost_sample:.0f}")
print(f"    需要毛利 > {op_cost_sample:.0f} 才能盈利")

