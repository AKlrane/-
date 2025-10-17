"""验证生产逻辑：购买原材料后是否会立即转化为产品"""
import sys
sys.path.append('.')

from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config
import numpy as np

print("="*80)
print("验证生产逻辑")
print("="*80)

# 预设的转换系数
conversion_ratios = {
    "Parts": {"input": "Raw", "ratio": 3, "desc": "3 Raw → 1 Parts"},
    "Electronics": {"input": "Raw", "ratio": 7, "desc": "7 Raw → 1 Electronics"},
    "Battery/Motor": {"input": "Raw", "ratio": 20, "desc": "20 Raw → 1 Battery/Motor"},
    "OEM": {"input": "Multiple", "ratios": {"Parts": 20, "Electronics": 10, "Battery/Motor": 4}, 
            "desc": "20 Parts OR 10 Electronics OR 4 Battery → 1 OEM"},
    "Service": {"input": "OEM", "ratio": 2, "desc": "2 OEM → 1 Service"}
}

print("\n1. 预设的转换系数:")
for sector, info in conversion_ratios.items():
    print(f"  {sector:15} {info['desc']}")

print("\n" + "="*80)
print("2. 验证生产流程（单个step）:")
print("="*80)

config = load_config("config/config.json")
env = IndustryEnv(config.environment)
env.reset(options={"initial_firms": 30})

# 找一个Parts公司来测试
parts_companies = [c for c in env.companies if sector_relations[c.sector_id].name == "Parts"]
if parts_companies:
    company = parts_companies[0]
    
    print(f"\n测试公司: Parts (ID={company.company_id})")
    print(f"  资本: {company.capital:.2f}")
    print(f"  生产能力比率: {company.production_capacity_ratio}")
    print(f"  最大生产量: {company.get_max_production():.2f}")
    
    # 模拟购买原材料
    initial_raw = company.raw_inventory
    print(f"\n  初始raw库存: {initial_raw:.2f}")
    
    # 手动给它一些raw库存
    company.raw_inventory = 300.0
    print(f"  设置raw库存: {company.raw_inventory:.2f}")
    
    # 计算理论上能生产多少
    max_from_raw = company.raw_inventory // 3
    max_from_capacity = company.get_max_production()
    
    print(f"\n  理论生产量:")
    print(f"    从原材料: {max_from_raw:.0f} (300 raw ÷ 3)")
    print(f"    从产能: {max_from_capacity:.0f} (capital × ratio)")
    
    # 执行生产
    initial_product = company.product_inventory
    produced = company.produce()
    final_product = company.product_inventory
    final_raw = company.raw_inventory
    
    print(f"\n  生产结果:")
    print(f"    生产数量: {produced:.0f}")
    print(f"    产品库存: {initial_product:.0f} → {final_product:.0f}")
    print(f"    原材料库存: 300.0 → {final_raw:.0f}")
    print(f"    消耗原材料: {300 - final_raw:.0f}")
    
    # 检查是否符合预期
    expected_produced = min(max_from_raw, max_from_capacity)
    expected_consumed = expected_produced * 3
    actual_consumed = 300 - final_raw
    
    if abs(produced - expected_produced) < 0.1 and abs(actual_consumed - expected_consumed) < 0.1:
        print(f"\n  ✅ 生产逻辑正确: 生产 {produced:.0f} 个，消耗 {actual_consumed:.0f} raw")
    else:
        print(f"\n  ❌ 生产逻辑异常!")
        print(f"     期望生产 {expected_produced:.0f}，实际 {produced:.0f}")
        print(f"     期望消耗 {expected_consumed:.0f}，实际 {actual_consumed:.0f}")
    
    # 检查是否有剩余
    if final_raw > 0:
        print(f"\n  ⚠️  剩余 {final_raw:.0f} raw 未转化 (受产能限制)")
        print(f"     如果想全部转化，需要增加产能或取消产能限制")

print("\n" + "="*80)
print("生产逻辑总结:")
print("="*80)
print("""
当前逻辑:
  1. 购买预算 = capital × tier_production_ratios
  2. 生产产能 = capital × tier_production_ratios（相同）
  3. 实际生产 = min(原材料能生产的量, 产能限制)

特点:
  ✅ 会尽力转化原材料为产品
  ✅ 按照预设的转换系数进行转化
  ⚠️  受产能限制，可能有原材料剩余到下一step
  
如果希望购买多少就100%转化多少:
  方案1: 确保 purchase_budget = production_capacity
  方案2: 去掉产能限制（生产量仅受原材料限制）
""")

