"""检查供应链网络是否正确建立"""
import sys
sys.path.append('.')

from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config

config = load_config("config/config.json")
env = IndustryEnv(config.environment)
env.reset(options={"initial_firms": 50})

print("="*80)
print("供应链网络检查")
print("="*80)

# 统计各sector的公司
sectors = {}
for c in env.companies:
    sector_name = sector_relations[c.sector_id].name
    if sector_name not in sectors:
        sectors[sector_name] = []
    sectors[sector_name].append(c)

print(f"\n公司分布:")
for sector_name, companies in sectors.items():
    if companies:
        tier = companies[0].tier
        print(f"  {sector_name:15} Tier {tier}: {len(companies)} 家公司")

# 检查Parts公司
print(f"\n{'='*80}")
print(f"检查Parts公司")
print(f"{'='*80}")

if "Parts" in sectors and sectors["Parts"]:
    parts = sectors["Parts"][0]
    print(f"\nParts公司示例:")
    print(f"  Tier: {parts.tier}")
    print(f"  供应商数量: {len(parts.suppliers)}")
    if parts.suppliers:
        sup_types = {}
        for sup in parts.suppliers:
            sup_name = sector_relations[sup.sector_id].name
            sup_types[sup_name] = sup_types.get(sup_name, 0) + 1
        print(f"  供应商类型: {sup_types}")
    
    print(f"  客户数量: {len(parts.customers)}")
    if parts.customers:
        cust_types = {}
        for cust in parts.customers:
            cust_name = sector_relations[cust.sector_id].name
            cust_types[cust_name] = cust_types.get(cust_name, 0) + 1
        print(f"  客户类型: {cust_types}")
    else:
        print(f"  ❌ 没有客户！Parts无法销售产品！")

# 检查OEM公司
print(f"\n{'='*80}")
print(f"检查OEM公司")
print(f"{'='*80}")

if "OEM" in sectors and sectors["OEM"]:
    oem = sectors["OEM"][0]
    print(f"\nOEM公司示例:")
    print(f"  Tier: {oem.tier}")
    print(f"  供应商数量: {len(oem.suppliers)}")
    if oem.suppliers:
        sup_types = {}
        for sup in oem.suppliers:
            sup_name = sector_relations[sup.sector_id].name
            sup_types[sup_name] = sup_types.get(sup_name, 0) + 1
        print(f"  供应商类型: {sup_types}")
        
        # 检查是否包含Parts
        has_parts = any(sector_relations[s.sector_id].name == "Parts" for s in oem.suppliers)
        has_elec = any(sector_relations[s.sector_id].name == "Electronics" for s in oem.suppliers)
        has_batt = any(sector_relations[s.sector_id].name == "Battery/Motor" for s in oem.suppliers)
        
        print(f"\n  供应商完整性:")
        print(f"    Parts: {'✅' if has_parts else '❌'}")
        print(f"    Electronics: {'✅' if has_elec else '❌'}")
        print(f"    Battery/Motor: {'✅' if has_batt else '❌'}")
    else:
        print(f"  ❌ 没有供应商！OEM无法采购！")
    
    print(f"  客户数量: {len(oem.customers)}")
    if oem.customers:
        cust_types = {}
        for cust in oem.customers:
            cust_name = sector_relations[cust.sector_id].name
            cust_types[cust_name] = cust_types.get(cust_name, 0) + 1
        print(f"  客户类型: {cust_types}")

# 检查完整的供应链流
print(f"\n{'='*80}")
print(f"完整供应链流检查")
print(f"{'='*80}")

print(f"\nRaw → Parts/Elec/Batt:")
if "Raw" in sectors and sectors["Raw"]:
    raw = sectors["Raw"][0]
    if raw.customers:
        cust_types = {}
        for cust in raw.customers:
            cust_name = sector_relations[cust.sector_id].name
            cust_types[cust_name] = cust_types.get(cust_name, 0) + 1
        print(f"  Raw有客户: {cust_types} ✅")
    else:
        print(f"  ❌ Raw没有客户")

print(f"\nParts/Elec/Batt → OEM:")
midstream_has_customers = False
for sector_name in ["Parts", "Electronics", "Battery/Motor"]:
    if sector_name in sectors and sectors[sector_name]:
        company = sectors[sector_name][0]
        if company.customers:
            midstream_has_customers = True
            oem_customers = sum(1 for c in company.customers if sector_relations[c.sector_id].name == "OEM")
            print(f"  {sector_name} → {oem_customers} OEM客户 ✅")
        else:
            print(f"  ❌ {sector_name}没有客户")

if not midstream_has_customers:
    print(f"\n🚨 问题确认：中游公司没有下游客户！")
    print(f"   这就是为什么Parts/Elec/Batt产品积压但无法销售")

print(f"\nOEM → Service:")
if "OEM" in sectors and sectors["OEM"]:
    oem = sectors["OEM"][0]
    if oem.customers:
        service_customers = sum(1 for c in oem.customers if sector_relations[c.sector_id].name == "Service")
        print(f"  OEM → {service_customers} Service客户 ✅")
    else:
        print(f"  ❌ OEM没有客户")

print(f"\n{'='*80}")
print(f"诊断结论")
print(f"{'='*80}")

# 检查tier设置
print(f"\nTier设置检查:")
from env.sector import SECTOR_TIERS
for sector_id, tier in SECTOR_TIERS.items():
    sector_name = sector_relations[sector_id].name
    print(f"  {sector_name:15} Tier {tier}")

