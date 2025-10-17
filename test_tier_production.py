"""测试按tier设置不同的生产能力"""
import sys
sys.path.append('.')

from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config

config = load_config("config/config.json")
env = IndustryEnv(config.environment)
env.reset(options={"initial_firms": 30})

print("="*80)
print("验证按tier的生产能力设置")
print("="*80)

# 按sector分组
sectors = {}
for company in env.companies:
    sector_name = sector_relations[company.sector_id].name
    if sector_name not in sectors:
        sectors[sector_name] = []
    sectors[sector_name].append(company)

# 打印每个sector的生产能力
print(f"\n配置的tier_production_ratios:")
for sector_name, ratio in config.environment.tier_production_ratios.items():
    print(f"  {sector_name}: {ratio}")

print(f"\n实际公司的生产能力比率:")
for sector_name, companies in sorted(sectors.items()):
    if companies:
        company = companies[0]
        ratio = company.production_capacity_ratio
        capital = company.capital
        max_production = company.get_max_production()
        print(f"\n  {sector_name}:")
        print(f"    production_capacity_ratio: {ratio}")
        print(f"    示例: 资本={capital:,.0f}, 最大生产={max_production:,.0f}")

# 检查供需平衡
print(f"\n" + "="*80)
print("供需平衡检查")
print("="*80)

raw_companies = sectors.get("Raw", [])
peb_companies = []
for name in ["Parts", "Electronics", "Battery/Motor"]:
    peb_companies.extend(sectors.get(name, []))

if raw_companies and peb_companies:
    raw_total_production = sum(c.get_max_production() for c in raw_companies)
    peb_total_demand = sum(c.get_max_purchase_budget() for c in peb_companies)
    
    print(f"\nRaw:")
    print(f"  公司数: {len(raw_companies)}")
    print(f"  总生产能力/步: {raw_total_production:,.0f}")
    print(f"  平均单个Raw生产: {raw_total_production/len(raw_companies):,.0f}")
    
    print(f"\nP/E/B:")
    print(f"  公司数: {len(peb_companies)}")
    print(f"  总采购预算/步: {peb_total_demand:,.0f}")
    print(f"  平均单个P/E/B采购: {peb_total_demand/len(peb_companies):,.0f}")
    
    ratio = peb_total_demand / max(raw_total_production, 1)
    print(f"\n供需比: {ratio:.2f}x (需求/供应)")
    
    if ratio > 2:
        print(f"  🔴 需求远大于供应！可能需要:")
        print(f"     - 增加Raw的production_capacity_ratio")
        print(f"     - 或减少P/E/B的purchase_budget_ratio")
    elif ratio < 0.5:
        print(f"  ⚠️ 供应远大于需求，Raw可能卖不出去")
    else:
        print(f"  ✅ 供需相对平衡")

print("\n" + "="*80)

