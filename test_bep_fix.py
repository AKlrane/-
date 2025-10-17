"""
验证 B/E/P 节点修复效果的简单测试
"""
import sys
sys.path.append('.')

import numpy as np
from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config

def test_bep_fix():
    print("=" * 70)
    print("测试 B/E/P 节点修复效果")
    print("=" * 70)
    
    # 加载环境
    config = load_config("config/config.json")
    env = IndustryEnv(config.environment)
    env.reset(options={"initial_firms": 30})
    
    # 找出 Parts 公司
    parts_companies = [c for c in env.companies if sector_relations[c.sector_id].name == "Parts"]
    
    if not parts_companies:
        print("❌ 没有找到 Parts 公司")
        return
    
    company = parts_companies[0]
    print(f"\n✓ 找到 Parts 公司")
    print(f"  初始资本: {company.capital:,.2f}")
    print(f"  采购预算: {company.get_max_purchase_budget():,.2f}")
    
    # 模拟5个步骤
    issues = []
    for step in range(5):
        capital_before = company.capital
        raw_before = company.raw_inventory
        
        # 执行步骤
        action = np.zeros(env.max_actions_per_step * 3, dtype=np.float32)
        env.step(action)
        
        capital_change = company.capital - capital_before
        raw_change = company.raw_inventory - raw_before
        
        print(f"\nStep {step + 1}:")
        print(f"  资本变化: {capital_change:+,.2f}")
        print(f"  原材料变化: {raw_change:+,.2f}")
        
        # 检查异常
        if raw_change > 50000:  # 原材料增长超过50k
            issues.append(f"Step {step+1}: 原材料爆炸增长 +{raw_change:,.2f}")
            print(f"  🔴 原材料爆炸增长！")
        
        if capital_change < -100000:  # 资本下降超过100k
            issues.append(f"Step {step+1}: 资本大幅下降 {capital_change:,.2f}")
            print(f"  🔴 资本大幅下降！")
        
        if company.capital < 0:
            issues.append(f"Step {step+1}: 资本变为负数")
            print(f"  🔴🔴🔴 资本变为负数！")
            break
    
    print("\n" + "=" * 70)
    if issues:
        print("❌ 测试失败，发现以下问题：")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ 测试通过！未发现明显异常")
    print("=" * 70)

if __name__ == "__main__":
    test_bep_fix()

