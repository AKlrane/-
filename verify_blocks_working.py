"""验证库存阻止机制是否真的在工作"""
import sys
sys.path.append('.')

import numpy as np
from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config

config = load_config("config/config.json")

print("="*80)
print("验证库存阻止机制（带详细追踪）")
print("="*80)

env = IndustryEnv(config.environment)
env._debug_inventory_blocks = True  # 启用调试模式
env.reset(options={"initial_firms": 50})

print(f"\n初始化完成：{len(env.companies)}家公司\n")

# 运行5步
for step_num in range(5):
    # 在step前记录状态
    pre_状态 = []
    for company in env.companies:
        if company.tier > 0:
            sector_name = sector_relations[company.sector_id].name
            inventory_value = company.product_inventory * company.revenue_rate
            inventory_ratio = inventory_value / max(company.capital, 1e-8)
            max_inventory_ratio = env.max_held_capital_rate.get(sector_name, 0.3)
            
            pre_状态.append({
                'company': company,
                'sector': sector_name,
                'pre_inv': company.product_inventory,
                'pre_ratio': inventory_ratio,
                'max_ratio': max_inventory_ratio,
                'should_block': inventory_ratio > max_inventory_ratio
            })
    
    # 重置调试计数
    env._inventory_blocks_count = {}
    
    # 执行step
    action = np.zeros(env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 检查结果
    print(f"Step {step_num + 1}:")
    print(f"  阻止统计: {env._inventory_blocks_count}")
    
    # 检查是否有应该被阻止但还是买了的
    bugs_found = []
    correct_blocks = []
    for state in pre_状态:
        company = state['company']
        did_purchase = company.products_purchased_this_step > 0
        
        if state['should_block'] and did_purchase:
            bugs_found.append(f"    ❌ {state['sector']:15} 应该阻止但买了 (比例={state['pre_ratio']:.1%} > {state['max_ratio']:.1%}, 买了{company.products_purchased_this_step:.1f})")
        elif state['should_block'] and not did_purchase:
            correct_blocks.append(state['sector'])
    
    if bugs_found:
        print(f"  ❌ 发现BUG:")
        for bug in bugs_found[:5]:  # 只显示前5个
            print(bug)
    else:
        print(f"  ✅ 没有发现BUG")
    
    if correct_blocks:
        print(f"  ✅ 正确阻止了 {len(correct_blocks)} 个公司")
    
    print()

print("="*80)
print("测试完成")
print("="*80)

