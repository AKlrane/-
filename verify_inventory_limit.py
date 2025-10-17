"""验证库存限制机制是否生效"""
import sys
sys.path.append('.')

import numpy as np
from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config

config = load_config("config/config.json")

print("="*80)
print("验证库存限制机制（各tier不同的max_held_capital_rate阈值）")
print("="*80)

print("\n【配置的max_held_capital_rate】")
if hasattr(config.environment, 'max_held_capital_rate'):
    for sector, rate in config.environment.max_held_capital_rate.items():
        print(f"  {sector:15} {rate:.1%}")
else:
    print("❌ 配置中没有max_held_capital_rate")

env = IndustryEnv(config.environment)
env.reset(options={"initial_firms": 50})

print(f"\n初始化完成：{len(env.companies)}家公司")

# 运行几步
for step in range(10):
    action = np.zeros(env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if step % 2 == 0:  # 每2步检查一次
        print(f"\n{'='*80}")
        print(f"Step {step + 1}:")
        print(f"{'='*80}")
        
        # 统计各tier的库存情况和是否购买
        tier_stats = {}
        
        for company in env.companies:
            sector_name = sector_relations[company.sector_id].name
            
            # 计算库存价值比例
            inventory_value = company.product_inventory * company.revenue_rate
            inventory_ratio = inventory_value / max(company.capital, 1e-8)
            
            # 获取该tier的阈值
            max_inventory_ratio = env.max_held_capital_rate.get(sector_name, 0.3)
            
            # 判断是否应该被限制购买
            should_skip = inventory_ratio > max_inventory_ratio
            
            # 检查是否实际进行了购买
            did_purchase = company.products_purchased_this_step > 0
            
            if sector_name not in tier_stats:
                tier_stats[sector_name] = {
                    'total': 0,
                    'high_inventory': 0,  # 库存比例>30%
                    'purchased': 0,       # 实际购买了
                    'blocked': 0,         # 应该被阻止且确实没买
                    'wrong_purchase': 0   # 应该被阻止但还是买了（bug）
                }
            
            stats = tier_stats[sector_name]
            stats['total'] += 1
            
            if inventory_ratio > 0.3:
                stats['high_inventory'] += 1
                if did_purchase:
                    stats['wrong_purchase'] += 1  # 这是bug！
                else:
                    stats['blocked'] += 1  # 正确阻止
            elif did_purchase:
                stats['purchased'] += 1
        
        # 输出统计
        for sector_name, stats in sorted(tier_stats.items()):
            if stats['total'] == 0:
                continue
            
            print(f"\n{sector_name}:")
            print(f"  总数: {stats['total']}")
            print(f"  高库存(>30%): {stats['high_inventory']}")
            print(f"  正常购买: {stats['purchased']}")
            print(f"  ✅ 正确阻止: {stats['blocked']}")
            
            if stats['wrong_purchase'] > 0:
                print(f"  ❌ 错误购买: {stats['wrong_purchase']} (应该被阻止但还是买了!)")
        
        # 输出几个高库存的例子
        print(f"\n【高库存公司示例】")
        count = 0
        for company in env.companies:
            sector_name = sector_relations[company.sector_id].name
            inventory_value = company.product_inventory * company.revenue_rate
            inventory_ratio = inventory_value / max(company.capital, 1e-8)
            max_inventory_ratio = env.max_held_capital_rate.get(sector_name, 0.3)
            
            if inventory_ratio > max_inventory_ratio:
                did_purchase = company.products_purchased_this_step > 0
                status = "❌买了" if did_purchase else "✅没买"
                print(f"  {sector_name:15} 库存价值={inventory_value:>8.0f} capital={company.capital:>8.0f} "
                      f"比例={inventory_ratio:>5.1%} (阈值={max_inventory_ratio:.1%}) {status}")
                count += 1
                if count >= 5:
                    break

print(f"\n{'='*80}")
print("验证完成！")
print(f"{'='*80}")

