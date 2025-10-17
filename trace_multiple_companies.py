"""同时追踪Parts、OEM、Service三家公司"""
import sys
sys.path.append('.')

from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config
import numpy as np

config = load_config("config/config.json")
env = IndustryEnv(config.environment)
obs, _ = env.reset(options={"initial_firms": 50})

print("="*80)
print("多公司追踪（Parts/OEM/Service）")
print("="*80)

# 选择追踪的公司
tracked = {}
for sector_name in ["Parts", "OEM", "Service"]:
    company = next((c for c in env.companies if sector_relations[c.sector_id].name == sector_name), None)
    if company:
        tracked[sector_name] = company
        print(f"追踪 {sector_name:10} 公司 (初始资本={company.capital:,.0f})")

if len(tracked) < 3:
    print("缺少公司!")
    exit()

print(f"\n运行20步...")

action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

for step in range(1, 21):
    print(f"\n{'='*80}")
    print(f"STEP {step}")
    print(f"{'='*80}")
    
    # 记录每家公司的状态
    before_states = {}
    for name, company in list(tracked.items()):
        if company in env.companies:
            before_states[name] = {
                'capital': company.capital,
                'product_inv': company.product_inventory,
                'sector_name': sector_relations[company.sector_id].name
            }
    
    # 执行step
    obs, reward, terminated, truncated, info = env.step(action)
    env.current_step = step
    
    # 分析每家公司
    for name, company in list(tracked.items()):
        if company not in env.companies:
            print(f"\n❌ {name} 已死亡!")
            del tracked[name]
            continue
        
        before = before_states[name]
        sector_name = sector_relations[company.sector_id].name
        
        print(f"\n【{name}】")
        print(f"  资本: {before['capital']:,.0f} → {company.capital:,.0f} ({company.capital - before['capital']:+,.0f})")
        print(f"  产品库存: {before['product_inv']:.0f} → {company.product_inventory:.0f}")
        
        # 显示sector特定的库存
        if sector_name in ("Parts", "Electronics", "Battery/Motor"):
            print(f"  Raw库存: {company.raw_inventory:.1f}")
        elif sector_name == "OEM":
            print(f"  Parts: {company.parts_inventory:.1f}, Elec: {company.electronics_inventory:.1f}, Batt: {company.battery_inventory:.1f}")
        elif sector_name == "Service":
            print(f"  OEM库存: {company.oem_inventory:.1f}")
        
        print(f"  活动: 采购={company.products_purchased_this_step:.0f}, 生产={company.products_produced_this_step:.0f}, 销售={company.products_sold_this_step:.0f}")
        
        # 计算营业利润（简化，不显示明细）
        if company.products_sold_this_step > 0:
            单位收入 = company.revenue_rate
            单位成本 = company.product_unit_cost
            毛利润 = company.products_sold_this_step * (单位收入 - 单位成本)
            print(f"  毛利: {毛利润:,.0f} (单位毛利={(单位收入 - 单位成本):.2f})")
        
        # 诊断问题
        capital_change = company.capital - before['capital']
        if capital_change < -1000:
            if company.products_sold_this_step == 0:
                print(f"  ⚠️  无销售，但有成本支出")
            else:
                print(f"  ⚠️  有销售但仍亏损，成本过高")
        
        # 检查供需
        if company.product_inventory > 10000:
            print(f"  ⚠️  库存积压 ({company.product_inventory:.0f})")
        if company.products_produced_this_step > company.products_sold_this_step * 2:
            print(f"  ⚠️  生产({company.products_produced_this_step:.0f}) >> 销售({company.products_sold_this_step:.0f})")
    
    # 显示整体统计
    print(f"\n【系统状态】")
    print(f"  总公司数: {len(env.companies)}")
    print(f"  总资本: {sum(c.capital for c in env.companies):,.0f}")
    
    # 如果所有追踪的公司都死了，结束
    if not tracked:
        print(f"\n所有追踪的公司都已死亡！")
        break

print(f"\n{'='*80}")
print(f"最终状态")
print(f"{'='*80}")

for name, company in tracked.items():
    if company in env.companies:
        print(f"\n{name}: {company.capital:,.0f} 资本")
    else:
        print(f"\n{name}: 已死亡")

print(f"\n系统: {len(env.companies)} 家公司存活, 总资本 {sum(c.capital for c in env.companies):,.0f}")

# 诊断总结
print(f"\n{'='*80}")
print(f"诊断总结")
print(f"{'='*80}")

print(f"\n观察到的问题:")
print(f"1. Parts能采购Raw吗？")
print(f"2. OEM能采购Parts吗？")  
print(f"3. Service能采购OEM吗？")
print(f"4. Parts能卖给OEM吗？")
print(f"5. OEM能卖给Service吗？")
print(f"6. Service能卖给市场吗？")
print(f"\n如果某个环节断了，整个供应链就会崩溃。")

