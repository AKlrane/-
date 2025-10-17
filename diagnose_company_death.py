"""诊断公司死亡原因 - 追踪详细的成本和收入"""
import sys
sys.path.append('.')

from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config
import numpy as np

print("="*80)
print("诊断公司死亡原因")
print("="*80)

config = load_config("config/config.json")
env = IndustryEnv(config.environment)
env.reset(options={"initial_firms": 50})

# 选择几家公司进行追踪
tracked_companies = {}
for sector_name in ["Raw", "Parts", "Electronics", "Battery/Motor", "OEM", "Service"]:
    company = next((c for c in env.companies if sector_relations[c.sector_id].name == sector_name), None)
    if company:
        tracked_companies[sector_name] = company
        print(f"追踪 {sector_name:15} 公司 (初始资本={company.capital:.0f})")

print("\n" + "="*80)
print("运行10步，详细追踪每一步")
print("="*80)

for step in range(1, 11):
    print(f"\n{'='*80}")
    print(f"STEP {step}")
    print(f"{'='*80}")
    
    # 记录step前的状态
    before_state = {}
    for name, company in tracked_companies.items():
        if company in env.companies:
            before_state[name] = {
                'capital': company.capital,
                'product_inv': company.product_inventory,
                'raw_inv': getattr(company, 'raw_inventory', 0),
                'parts_inv': getattr(company, 'parts_inventory', 0),
                'elec_inv': getattr(company, 'electronics_inventory', 0),
                'batt_inv': getattr(company, 'battery_inventory', 0),
            }
    
    # 执行一步
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, reward, done, truncated, info = env.step(action)
    
    # 分析变化
    for name, company in list(tracked_companies.items()):
        if company not in env.companies:
            print(f"\n❌ {name} 已死亡!")
            del tracked_companies[name]
            continue
        
        before = before_state[name]
        
        print(f"\n【{name}】")
        print(f"  资本: {before['capital']:.0f} → {company.capital:.0f} "
              f"({company.capital - before['capital']:+.0f})")
        
        # 库存变化
        print(f"  产品库存: {before['product_inv']:.1f} → {company.product_inventory:.1f}")
        if name in ("Parts", "Electronics", "Battery/Motor"):
            print(f"  raw库存: {before['raw_inv']:.1f} → {company.raw_inventory:.1f}")
        elif name == "OEM":
            print(f"  parts库存: {before['parts_inv']:.1f} → {company.parts_inventory:.1f}")
            print(f"  elec库存: {before['elec_inv']:.1f} → {company.electronics_inventory:.1f}")
            print(f"  batt库存: {before['batt_inv']:.1f} → {company.battery_inventory:.1f}")
        
        # 本步统计
        print(f"  生产: {company.products_produced_this_step:.1f}")
        print(f"  销售: {company.products_sold_this_step:.1f}")
        print(f"  采购: {company.products_purchased_this_step:.1f}")
        
        # 财务详情（这些在step()执行后已清零，需要从资本变化推断）
        capital_change = company.capital - before['capital']
        print(f"  资本变化: {capital_change:+.0f}")
        
        # 计算预算和成本
        purchase_budget = company.get_max_purchase_budget()
        production_capacity = company.get_max_production()
        op_cost_estimate = company.op_cost_rate * before['capital']
        
        print(f"  采购预算: {purchase_budget:.0f} (capital × {company.production_capacity_ratio})")
        print(f"  生产能力: {production_capacity:.0f}")
        print(f"  运营成本估计: {op_cost_estimate:.0f}")

print("\n" + "="*80)
print("关键指标分析")
print("="*80)

for name, company in tracked_companies.items():
    if company in env.companies:
        sector = sector_relations[company.sector_id]
        print(f"\n{name}:")
        print(f"  Tier: {company.tier}")
        print(f"  资本: {company.capital:.0f}")
        print(f"  采购比率: {company.production_capacity_ratio}")
        print(f"  运营成本倍率: {sector.operating_cost_multiplier}")
        print(f"  收入率(单价): {company.revenue_rate:.2f}")
        print(f"  产品库存: {company.product_inventory:.1f}")
        print(f"  供应商数量: {len(company.suppliers)}")
        print(f"  客户数量: {len([c for c in env.companies if company in c.suppliers])}")

print("\n" + "="*80)
print("可能的问题诊断")
print("="*80)
print("""
请检查以下几点:
1. 中游公司(Parts/Elec/Batt)是否能够购买到足够的Raw材料？
2. 购买的材料是否能及时转化为产品？（受产能限制）
3. 生产的产品是否能卖出去？（OEM是否有足够预算购买）
4. 运营成本是否过高导致入不敷出？
5. 价格体系是否合理？（Raw→Parts→OEM→Service的价格链）

建议:
- 如果产能限制导致材料积压，可以提高tier_production_ratios
- 如果卖不出去，可能是下游预算不足或价格太高
- 如果成本过高，可以调整op_cost_rate或sector的cost_multiplier
""")
