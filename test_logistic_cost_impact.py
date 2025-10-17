"""测试logistic_cost_rate是否真正影响运输成本"""
import sys
sys.path.append('.')

import numpy as np
from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config

# 测试两种配置
print("="*80)
print("测试logistic_cost_rate的影响")
print("="*80)

config = load_config("config/config.json")

print(f"\n【当前配置】")
print(f"  logistic_cost_rate: {config.environment.logistic_cost_rate}")
print(f"  disable_logistic_costs: {config.environment.disable_logistic_costs}")

env = IndustryEnv(config.environment)
print(f"\n【环境加载后】")
print(f"  env.logistic_cost_rate: {env.logistic_cost_rate}")
print(f"  env.disable_logistic_costs: {env.disable_logistic_costs}")

env.reset(options={"initial_firms": 30}, seed=42)

# 检查公司的logistic_cost_rate
if env.companies:
    sample = env.companies[0]
    print(f"\n【公司对象】")
    print(f"  公司的logistic_cost_rate: {sample.logistic_cost_rate}")

# 运行几步，追踪运输成本
print(f"\n{'='*80}")
print("运行模拟，追踪运输成本支出")
print(f"{'='*80}")

total_logistic_costs = []

for step in range(10):
    # 记录step前的capital
    capital_before = {id(c): c.capital for c in env.companies}
    
    action = np.zeros(env.action_space.shape)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # 计算运输成本（通过capital变化）
    # 这里我们需要手动计算，因为运输成本直接从capital扣除
    
    if step % 3 == 0:
        # 统计本步的购买和可能的运输成本
        buyers = [c for c in env.companies if c.products_purchased_this_step > 0]
        
        print(f"\nStep {step + 1}:")
        print(f"  购买公司数: {len(buyers)}")
        
        if buyers:
            # 采样一些买家
            for i, buyer in enumerate(buyers[:3]):
                sector_name = sector_relations[buyer.sector_id].name
                capital_change = buyer.capital - capital_before.get(id(buyer), buyer.capital)
                
                print(f"    买家#{i+1} ({sector_name}):")
                print(f"      购买量: {buyer.products_purchased_this_step:.1f}")
                print(f"      Revenue: {buyer.revenue:.0f}")
                print(f"      Capital变化: {capital_change:,.0f}")
                
                # 尝试找到供应商
                if buyer.suppliers:
                    nearest = sorted(buyer.suppliers, key=lambda s: buyer.distance_to(s))[:2]
                    for j, supp in enumerate(nearest):
                        dist = buyer.distance_to(supp)
                        supp_sector = sector_relations[supp.sector_id].name
                        # 估算运输成本
                        if supp.products_sold_this_step > 0:
                            unit_price = supp.revenue_rate
                            # 假设这个买家从这个供应商买了一些
                            est_units = buyer.products_purchased_this_step / len(nearest)
                            est_logistic = supp.logistic_cost_rate * unit_price * est_units * dist
                            print(f"      供应商#{j+1} ({supp_sector}): 距离={dist:.1f}, 估算运输成本≈{est_logistic:,.0f}")

print(f"\n{'='*80}")
print("结论")
print(f"{'='*80}")
print(f"\n如果logistic_cost_rate正常工作：")
print(f"  - 距离远的交易应该有更高的运输成本")
print(f"  - 调高logistic_cost_rate应该导致capital下降更快")
print(f"\n当前logistic_cost_rate = {config.environment.logistic_cost_rate}")
print(f"\n运输成本公式: logistic_cost_rate × 单价 × 数量 × 距离")
print(f"例如：0.05 × 180(OEM) × 10 × 20(距离) = 1,800")
print(f"\n如果改成1.0，同样的交易成本 = 36,000（20倍！）")

print(f"\n{'='*80}")

