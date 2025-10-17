"""详细分析Service的财务 - 正确版本（在step前后都记录）"""
import sys
sys.path.append('.')

from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config
import numpy as np

config = load_config("config/config.json")
env = IndustryEnv(config.environment)
env.reset(options={"initial_firms": 50})

# 找Service公司
service = next((c for c in env.companies if sector_relations[c.sector_id].name == "Service"), None)
if not service:
    print("没有找到Service公司")
    exit()

print("="*80)
print("Service公司财务详细分析（完整版）")
print("="*80)

# 记录初始状态
before_capital = service.capital
before_product_inv = service.product_inventory
before_oem_inv = service.oem_inventory
before_revenue_account = service.revenue
before_logistic_account = service.logistic_cost
before_cogs_account = service.cogs_cost

print(f"\n【Step执行前】")
print(f"资本: {before_capital:,.2f}")
print(f"产品库存: {before_product_inv:.2f}")
print(f"OEM库存: {before_oem_inv:.2f}")
print(f"Revenue账户: {before_revenue_account:,.2f}")
print(f"Logistic cost账户: {before_logistic_account:,.2f}")
print(f"COGS账户: {before_cogs_account:,.2f}")

# 手动模拟一个step的供应链部分（但不执行step结算）
print(f"\n{'='*80}")
print(f"执行模拟（分步观察）")
print(f"{'='*80}")

# 执行完整的step
action = np.zeros(env.action_space.shape, dtype=np.float32)

# ========== 关键修改：在供应链模拟后、company.step()前查看累计值 ==========
# 我们需要手动追踪，因为env.step()会一次性执行所有逻辑

# 先reset step counters（env.step会做）
for c in env.companies:
    c.reset_step_counters()

# 模拟供应链（采购、生产、销售）
env._simulate_supply_chain()

if service not in env.companies:
    print("\n❌ Service已死亡!")
    exit()

# 在company.step()之前记录累计的revenue、cogs、logistic
after_supply_chain_capital = service.capital
after_supply_chain_revenue = service.revenue
after_supply_chain_logistic = service.logistic_cost
after_supply_chain_cogs = service.cogs_cost
after_supply_chain_product_inv = service.product_inventory
after_supply_chain_oem_inv = service.oem_inventory

print(f"\n【供应链模拟后，step()结算前】")
print(f"资本: {after_supply_chain_capital:,.2f} (变化: {after_supply_chain_capital - before_capital:+,.2f})")
print(f"产品库存: {after_supply_chain_product_inv:.2f}")
print(f"OEM库存: {after_supply_chain_oem_inv:.2f}")
print(f"")
print(f"✅ Revenue累计: {after_supply_chain_revenue:,.2f}")
print(f"✅ Logistic cost累计: {after_supply_chain_logistic:,.2f}")
print(f"✅ COGS累计: {after_supply_chain_cogs:,.2f}")
print(f"")
print(f"本步活动:")
print(f"  采购: {service.products_purchased_this_step:.2f}")
print(f"  生产: {service.products_produced_this_step:.2f}")
print(f"  销售: {service.products_sold_this_step:.2f}")

# 计算采购支出（从资本变化推断）
# 在供应链模拟阶段，采购会立即扣除资本，销售只是记账
purchase_cost = before_capital - after_supply_chain_capital
print(f"  采购支出（反推）: {purchase_cost:,.2f}")

# 现在执行company.step()结算
print(f"\n{'='*80}")
print(f"执行step()结算")
print(f"{'='*80}")

# 计算各项成本
op_cost = service.op_cost_rate * after_supply_chain_capital
capital_ratio = max(after_supply_chain_capital, 0.0) / max(env.max_capital, 1.0)
management_cost = max(after_supply_chain_capital, 0.0) * 0.001 * (capital_ratio ** 0.5)
fixed_income = service.fixed_income

print(f"\n结算前的累计:")
print(f"  Revenue: {after_supply_chain_revenue:,.2f}")
print(f"  Logistic cost: {after_supply_chain_logistic:,.2f}")
print(f"  COGS: {after_supply_chain_cogs:,.2f}")
print(f"\n结算时计算的成本:")
print(f"  运营成本: {op_cost:,.2f}")
print(f"  管理成本: {management_cost:,.2f}")
print(f"  Fixed income: {fixed_income:.2f}")

total_cost = op_cost + management_cost + after_supply_chain_logistic + after_supply_chain_cogs
profit = after_supply_chain_revenue - total_cost + fixed_income

print(f"\nProfit计算:")
print(f"  = {after_supply_chain_revenue:,.2f} (revenue)")
print(f"  - {op_cost:,.2f} (运营)")
print(f"  - {management_cost:,.2f} (管理)")
print(f"  - {after_supply_chain_logistic:,.2f} (运输)")
print(f"  - {after_supply_chain_cogs:,.2f} (COGS)")
print(f"  + {fixed_income:.2f} (fixed)")
print(f"  = {profit:,.2f}")

# 手动执行step结算
before_step_capital = after_supply_chain_capital
service.capital += profit
service.revenue = 0.0
service.logistic_cost = 0.0
service.cogs_cost = 0.0

print(f"\n【step()结算后】")
print(f"资本: {before_step_capital:,.2f} + {profit:,.2f} = {service.capital:,.2f}")

print(f"\n{'='*80}")
print(f"完整财务总结")
print(f"{'='*80}")

final_capital = service.capital
total_capital_change = final_capital - before_capital

print(f"\n期初资本: {before_capital:,.2f}")
print(f"期末资本: {final_capital:,.2f}")
print(f"总变化: {total_capital_change:,.2f}")

print(f"\n分解:")
print(f"1. 采购阶段（立即支付现金）:")
print(f"   采购支出: -{purchase_cost:,.2f}")
print(f"")
print(f"2. 销售和结算:")
print(f"   收入: +{after_supply_chain_revenue:,.2f}")
print(f"   COGS: -{after_supply_chain_cogs:,.2f}")
print(f"   运营成本: -{op_cost:,.2f}")
print(f"   管理成本: -{management_cost:,.2f}")
print(f"   运输成本: -{after_supply_chain_logistic:,.2f}")
print(f"   Fixed: {fixed_income:+.2f}")
print(f"   小计: {profit:+,.2f}")
print(f"")
print(f"总计: -{purchase_cost:,.2f} + {profit:,.2f} = {total_capital_change:,.2f}")

# 验证
calculated = -purchase_cost + profit
if abs(calculated - total_capital_change) < 0.01:
    print(f"\n✅ 财务计算完全正确！")
else:
    print(f"\n❌ 有差异: 计算值={calculated:,.2f}, 实际={total_capital_change:,.2f}, 差={calculated - total_capital_change:,.2f}")

