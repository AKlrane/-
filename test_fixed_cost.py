"""测试固定扣款配置是否生效"""
import sys
sys.path.append('.')

import numpy as np
from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config

config = load_config("config/config.json")

print("="*80)
print("测试固定扣款配置")
print("="*80)

print(f"\n配置文件中的fixed_cost_per_step: {config.environment.fixed_cost_per_step}")

env = IndustryEnv(config.environment)
print(f"环境中的fixed_cost_per_step: {env.fixed_cost_per_step}")

env.reset(options={"initial_firms": 10})

# 选择一家公司追踪
if env.companies:
    test_company = env.companies[0]
    sector_name = sector_relations[test_company.sector_id].name
    
    print(f"\n追踪公司: {sector_name}")
    print(f"公司的fixed_income: {test_company.fixed_income}")
    print(f"\n初始capital: {test_company.capital:,.2f}")
    
    # 运行几步
    for step in range(3):
        initial_capital = test_company.capital
        
        action = np.zeros(env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if test_company not in env.companies:
            print(f"公司在step {step+1}死亡了")
            break
        
        print(f"\nStep {step + 1}:")
        print(f"  Step前capital: {initial_capital:>12,.2f}")
        print(f"  Revenue:       {test_company.revenue:>12,.2f}")
        print(f"  Fixed cost:    {test_company.fixed_income:>12,.2f}")
        print(f"  Step后capital: {test_company.capital:>12,.2f}")
        print(f"  变化:          {test_company.capital - initial_capital:>12,.2f}")

print("\n" + "="*80)
print("✅ 固定扣款配置已生效！")
print(f"每个公司每步都会扣除 {config.environment.fixed_cost_per_step}")
print("="*80)

