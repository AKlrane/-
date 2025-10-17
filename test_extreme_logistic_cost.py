"""测试极端logistic_cost_rate是否真的生效"""
import sys
sys.path.append('.')

import numpy as np
from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config

print("="*80)
print("测试极端logistic_cost_rate")
print("="*80)

# 先清除缓存确保使用最新配置
import os
import shutil
for cache_dir in ['config/__pycache__', 'env/__pycache__']:
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"清除缓存: {cache_dir}")

# 重新加载
config = load_config("config/config.json")

print(f"\n【配置文件】")
print(f"  logistic_cost_rate: {config.environment.logistic_cost_rate}")
print(f"  death_threshold: {config.environment.death_threshold}")

env = IndustryEnv(config.environment)

print(f"\n【环境】")
print(f"  env.logistic_cost_rate: {env.logistic_cost_rate}")

env.reset(options={"initial_firms": 30}, seed=42)

# 检查所有公司的logistic_cost_rate
rates = [c.logistic_cost_rate for c in env.companies]
print(f"\n【公司的logistic_cost_rate】")
print(f"  所有公司的值: {set(rates)}")
print(f"  是否一致: {'✅' if len(set(rates)) == 1 else '❌'}")

if len(set(rates)) == 1 and list(set(rates))[0] == config.environment.logistic_cost_rate:
    print(f"  ✅ 配置正确传递到公司！")
else:
    print(f"  ❌ 配置没有正确传递！")
    print(f"  期望值: {config.environment.logistic_cost_rate}")
    print(f"  实际值: {set(rates)}")

# 运行几步看看
print(f"\n{'='*80}")
print("运行5步，观察公司死亡情况")
print(f"{'='*80}")

initial_counts = {
    sector_relations[c.sector_id].name: sum(1 for co in env.companies if co.sector_id == c.sector_id)
    for c in env.companies
}

for step in range(5):
    action = np.zeros(env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    
    alive = len(env.companies)
    
    # 统计购买者
    buyers = [c for c in env.companies if c.products_purchased_this_step > 0]
    
    # 找到capital最低的几个
    sorted_by_capital = sorted(env.companies, key=lambda c: c.capital)
    
    print(f"\nStep {step + 1}:")
    print(f"  存活公司: {alive}")
    print(f"  购买者: {len(buyers)}")
    
    if buyers:
        # 检查一个买家的实际运输成本扣款
        buyer = buyers[0]
        sector_name = sector_relations[buyer.sector_id].name
        
        print(f"  示例买家 ({sector_name}):")
        print(f"    Capital: {buyer.capital:,.0f}")
        print(f"    购买量: {buyer.products_purchased_this_step:.1f}")
        print(f"    logistic_cost_rate: {buyer.logistic_cost_rate}")
        
        # 如果有供应商，计算理论运输成本
        if buyer.suppliers:
            nearest = sorted(buyer.suppliers, key=lambda s: buyer.distance_to(s))[0]
            dist = buyer.distance_to(nearest)
            unit_price = nearest.revenue_rate
            
            # 假设购买均分
            est_units = buyer.products_purchased_this_step / min(2, len(buyer.suppliers))
            theoretical_cost = buyer.logistic_cost_rate * unit_price * est_units * dist
            
            print(f"    最近供应商距离: {dist:.1f}")
            print(f"    理论运输成本: {theoretical_cost:,.0f}")
            
            if buyer.logistic_cost_rate > 1000:
                print(f"    ⚠️  如果logistic_cost_rate={buyer.logistic_cost_rate:.0f}，运输成本应该是天文数字！")
    
    # 显示最低capital的公司
    if sorted_by_capital:
        print(f"  最低capital的3家:")
        for i in range(min(3, len(sorted_by_capital))):
            c = sorted_by_capital[i]
            sector_name = sector_relations[c.sector_id].name
            print(f"    {sector_name}: {c.capital:,.0f}")

print(f"\n{'='*80}")
print("结论")
print(f"{'='*80}")

final_counts = {
    sector_relations[c.sector_id].name: sum(1 for co in env.companies if co.sector_id == c.sector_id)
    for c in env.companies
}

print(f"\n各tier存活情况:")
for sector in ["Raw", "Parts", "Electronics", "Battery/Motor", "OEM", "Service"]:
    initial = initial_counts.get(sector, 0)
    final = final_counts.get(sector, 0)
    died = initial - final
    print(f"  {sector:15}: {initial} → {final} (死亡{died})")

if config.environment.logistic_cost_rate > 1000:
    print(f"\n⚠️  警告：logistic_cost_rate = {config.environment.logistic_cost_rate:.0e}")
    print(f"如果配置正确生效，中游公司应该瞬间死光！")
    if sum(died for died in [initial_counts.get(s, 0) - final_counts.get(s, 0) for s in ["Parts", "Electronics", "Battery/Motor", "OEM"]]) == 0:
        print(f"❌ 中游没有死亡，配置可能没有生效！")
    else:
        print(f"✅ 配置生效了！")

print(f"\n{'='*80}")

