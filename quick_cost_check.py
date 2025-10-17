"""快速成本检查 - 只模拟20步"""
import sys
sys.path.append('.')

import numpy as np
from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config

config = load_config("config/config.json")
env = IndustryEnv(config.environment)
env.reset(options={"initial_firms": 50})

print("初始公司数:", len(env.companies))

# 找一个Parts公司和一个OEM公司
parts = next((c for c in env.companies if sector_relations[c.sector_id].name == "Parts"), None)
oem = next((c for c in env.companies if sector_relations[c.sector_id].name == "OEM"), None)

if not parts:
    print("没找到Parts公司")
    sys.exit(1)
if not oem:
    print("没找到OEM公司")
    sys.exit(1)

print(f"\nParts公司: 资本={parts.capital:,.0f}, 位置=({parts.x:.1f},{parts.y:.1f})")
print(f"OEM公司: 资本={oem.capital:,.0f}, 位置=({oem.x:.1f},{oem.y:.1f})")

# 模拟20步
action = np.zeros(env.max_actions_per_step * 3, dtype=np.float32)

print("\n" + "="*80)
for step in range(1, 21):
    # Parts公司
    if parts in env.companies:
        p_cap_before = parts.capital
        p_rev_before = parts.revenue
        p_logistic_before = parts.logistic_cost
        p_cogs_before = parts.cogs_cost
    
    # OEM公司
    if oem in env.companies:
        o_cap_before = oem.capital
        o_rev_before = oem.revenue
        o_logistic_before = oem.logistic_cost
        o_cogs_before = oem.cogs_cost
    
    # 执行步骤
    env.step(action)
    
    print(f"\nStep {step}: 总公司数={len(env.companies)}")
    
    # 报告Parts
    if parts in env.companies:
        p_profit = parts.capital - p_cap_before
        p_op_cost = parts.op_cost_rate * p_cap_before
        print(f"  Parts: 资本={parts.capital:>10,.0f} (变化:{p_profit:>+8,.0f})")
        print(f"    收入={p_rev_before:>8,.0f}, 运营={p_op_cost:>8,.0f}, 物流={p_logistic_before:>8,.0f}, COGS={p_cogs_before:>8,.0f}")
        print(f"    销售={parts.products_sold_this_step:.0f}, 生产={parts.products_produced_this_step:.0f}")
    else:
        print(f"  Parts: 💀 已死亡")
    
    # 报告OEM
    if oem in env.companies:
        o_profit = oem.capital - o_cap_before
        o_op_cost = oem.op_cost_rate * o_cap_before
        print(f"  OEM:   资本={oem.capital:>10,.0f} (变化:{o_profit:>+8,.0f})")
        print(f"    收入={o_rev_before:>8,.0f}, 运营={o_op_cost:>8,.0f}, 物流={o_logistic_before:>8,.0f}, COGS={o_cogs_before:>8,.0f}")
        print(f"    销售={oem.products_sold_this_step:.0f}, 生产={oem.products_produced_this_step:.0f}")
    else:
        print(f"  OEM:   💀 已死亡")
    
    if parts not in env.companies and oem not in env.companies:
        print("\n两个公司都死了，停止模拟")
        break

print("\n" + "="*80)
print(f"最终公司数: {len(env.companies)}")

