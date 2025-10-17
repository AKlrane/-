"""详细分析Service的财务状况"""
import sys
sys.path.append('.')

from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config
import numpy as np

config = load_config("config/config.json")
env = IndustryEnv(config.environment)
env.reset(options={"initial_firms": 50})

# 找一个Service公司
service = next((c for c in env.companies if sector_relations[c.sector_id].name == "Service"), None)

if not service:
    print("没有找到Service公司")
    exit()

print("="*80)
print("Service公司财务详细分析")
print("="*80)

# 记录初始状态
print(f"\n初始状态:")
print(f"  资本: {service.capital:,.2f}")
print(f"  收入率(单价): {service.revenue_rate:.2f}")
print(f"  运营成本率: {service.op_cost_rate:.6f}")
print(f"  产品库存: {service.product_inventory:.2f}")
print(f"  OEM库存: {service.oem_inventory:.2f}")
print(f"  OEM单位成本: {service.input_cost_per_unit.get('oem', 0):.2f}")
print(f"  产品单位成本: {service.product_unit_cost:.2f}")

# 运行一步
print(f"\n{'='*80}")
print(f"执行一步模拟")
print(f"{'='*80}")

before_capital = service.capital
before_product_inv = service.product_inventory
before_oem_inv = service.oem_inventory
before_revenue = service.revenue
before_logistic = service.logistic_cost
before_cogs = service.cogs_cost

action = np.zeros(env.action_space.shape, dtype=np.float32)
obs, reward, done, truncated, info = env.step(action)

if service not in env.companies:
    print("\n❌ Service公司已死亡!")
    exit()

print(f"\nStep后状态:")
print(f"  资本: {before_capital:,.2f} → {service.capital:,.2f} ({service.capital - before_capital:+,.2f})")
print(f"  产品库存: {before_product_inv:.2f} → {service.product_inventory:.2f}")
print(f"  OEM库存: {before_oem_inv:.2f} → {service.oem_inventory:.2f}")

print(f"\n本步活动:")
print(f"  采购: {service.products_purchased_this_step:.2f} 个OEM")
print(f"  生产: {service.products_produced_this_step:.2f} 个Service")
print(f"  销售: {service.products_sold_this_step:.2f} 个Service")

print(f"\n财务计算（step()结算前的累计）:")
print(f"  Revenue累计: {before_revenue:,.2f}")
print(f"  Logistic cost累计: {before_logistic:,.2f}")
print(f"  COGS累计: {before_cogs:,.2f}")

# 计算各项成本
op_cost = service.op_cost_rate * before_capital
print(f"\n成本明细:")
print(f"  1. 运营成本 = {before_capital:,.2f} × {service.op_cost_rate:.6f} = {op_cost:,.2f}")

capital_ratio = max(before_capital, 0.0) / max(env.max_capital, 1.0)
management_cost = max(before_capital, 0.0) * 0.001 * (capital_ratio ** 0.5)
print(f"  2. 管理成本 = {before_capital:,.2f} × 0.001 × {capital_ratio**0.5:.6f} = {management_cost:,.2f}")

print(f"  3. 运输成本(累计): {before_logistic:,.2f}")
print(f"  4. COGS(累计): {before_cogs:,.2f}")
print(f"  5. Fixed income: {service.fixed_income:.2f}")

# 计算收入
print(f"\n收入明细:")
units_sold = service.products_sold_this_step
revenue_calculated = units_sold * service.revenue_rate
print(f"  销售: {units_sold:.2f} 个 × {service.revenue_rate:.2f} = {revenue_calculated:,.2f}")
print(f"  实际revenue: {before_revenue:,.2f}")

# 计算COGS
print(f"\nCOGS计算:")
print(f"  产品单位成本: {service.product_unit_cost:.2f}")
print(f"  销售数量: {units_sold:.2f}")
print(f"  COGS = {units_sold:.2f} × {service.product_unit_cost:.2f} = {units_sold * service.product_unit_cost:,.2f}")
print(f"  实际COGS: {before_cogs:,.2f}")

# 计算采购支出
oem_price = 205.0  # 从config读取
purchased = service.products_purchased_this_step
purchase_cost = purchased * oem_price
print(f"\n采购支出:")
print(f"  采购: {purchased:.2f} 个OEM × {oem_price:.2f} = {purchase_cost:,.2f}")

# 总结
print(f"\n{'='*80}")
print(f"财务总结")
print(f"{'='*80}")

total_cost = op_cost + management_cost + before_logistic + before_cogs
profit_from_step = before_revenue - total_cost + service.fixed_income
capital_change_from_step = profit_from_step
capital_change_from_purchase = -purchase_cost
total_capital_change = capital_change_from_step + capital_change_from_purchase
actual_capital_change = service.capital - before_capital

print(f"\nStep结算:")
print(f"  收入: {before_revenue:,.2f}")
print(f"  - 运营成本: {op_cost:,.2f}")
print(f"  - 管理成本: {management_cost:,.2f}")
print(f"  - 运输成本: {before_logistic:,.2f}")
print(f"  - COGS: {before_cogs:,.2f}")
print(f"  + Fixed income: {service.fixed_income:.2f}")
print(f"  = Profit: {profit_from_step:,.2f}")

print(f"\n资本变化:")
print(f"  采购支出: -{purchase_cost:,.2f}")
print(f"  Step profit: +{profit_from_step:,.2f}")
print(f"  理论总变化: {total_capital_change:,.2f}")
print(f"  实际变化: {actual_capital_change:,.2f}")
print(f"  差异: {actual_capital_change - total_capital_change:,.2f}")

# 分析为什么亏钱
print(f"\n{'='*80}")
print(f"亏损原因分析")
print(f"{'='*80}")

if actual_capital_change < 0:
    print(f"\n❌ 亏损 {-actual_capital_change:,.2f}")
    print(f"\n主要原因:")
    
    if before_cogs > before_revenue * 0.5:
        print(f"  ⚠️  COGS占收入 {before_cogs/before_revenue*100:.1f}% (太高!)")
    
    if before_logistic > before_revenue * 0.1:
        print(f"  ⚠️  运输成本占收入 {before_logistic/before_revenue*100:.1f}% (可能过高)")
    
    if purchase_cost > before_revenue:
        print(f"  ⚠️  采购支出 {purchase_cost:,.2f} > 收入 {before_revenue:,.2f}")
        print(f"     说明采购过多，资本被占用")
else:
    print(f"\n✅ 盈利 {actual_capital_change:,.2f}")

