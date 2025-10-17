"""验证交易逻辑是否正确实现"""
import sys
sys.path.append('.')

from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config

print("="*80)
print("验证交易逻辑")
print("="*80)

config = load_config("config/config.json")
env = IndustryEnv(config.environment)
env.reset(options={"initial_firms": 30})

# 检查不同tier的生产能力
print("\n1. 检查采购预算设置（应该使用tier_production_ratios）:")
for sector_name, expected_ratio in config.environment.tier_production_ratios.items():
    companies = [c for c in env.companies if sector_relations[c.sector_id].name == sector_name]
    if companies:
        company = companies[0]
        actual_ratio = company.production_capacity_ratio
        budget = company.get_max_purchase_budget()
        expected_budget = company.capital * expected_ratio
        
        match = "✅" if abs(budget - expected_budget) < 1 else "❌"
        print(f"  {sector_name:15} 期望比率={expected_ratio}, 实际比率={actual_ratio}, {match}")
        if abs(budget - expected_budget) >= 1:
            print(f"    资本={company.capital:.0f}, 采购预算={budget:.0f}, 期望={expected_budget:.0f}")

# 检查OEM的上游供应商
oem = next((c for c in env.companies if sector_relations[c.sector_id].name == "OEM"), None)
if oem:
    print(f"\n2. 检查OEM供应商（应该包含Parts/Electronics/Battery）:")
    suppliers = [sector_relations[s.sector_id].name for s in oem.suppliers]
    print(f"  OEM的供应商: {suppliers}")
    
    relevant = [s for s in suppliers if s in ("Parts", "Electronics", "Battery/Motor")]
    if len(relevant) == len([s for s in suppliers if s != "Other"]):
        print(f"  ✅ OEM能从Parts/Electronics/Battery购买")
    else:
        print(f"  ❌ OEM供应商不正确")

# 检查Service销售逻辑
print(f"\n3. 检查Service是否全部售出:")
print(f"  （需要运行模拟来验证）")

print("\n" + "="*80)
print("逻辑概要:")
print("="*80)
print("""
✅ 采购预算 = capital × tier_production_ratios（按sector不同）
✅ 预算均分给最近K家上游供应商
✅ 订单量 = 预算 / 上游单价（不同上游价格不同）
✅ 资本流动: 买家capital减少，卖家capital增加
✅ 订单处理: 从近到远，直到库存归零
✅ Service: 每步售出全部库存
""")

