"""
详细跟踪单个Parts公司的交易细节
"""
import sys
sys.path.append('.')

import numpy as np
from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config

def patch_env_to_track_transactions(env, target_company_idx):
    """
    Monkey-patch环境的_simulate_supply_chain方法来记录交易
    """
    original_simulate = env._simulate_supply_chain
    transactions = []
    
    def tracked_simulate():
        # 记录目标公司在交易前的状态
        if target_company_idx < len(env.companies):
            company = env.companies[target_company_idx]
            before_state = {
                'capital': company.capital,
                'raw_inventory': company.raw_inventory,
                'product_inventory': company.product_inventory,
            }
        
        # 先调用原始方法
        original_simulate()
        
        # 记录交易后的状态
        if target_company_idx < len(env.companies):
            company = env.companies[target_company_idx]
            after_state = {
                'capital': company.capital,
                'raw_inventory': company.raw_inventory,
                'product_inventory': company.product_inventory,
            }
            
            transactions.append({
                'before': before_state,
                'after': after_state,
                'purchased': company.products_purchased_this_step,
                'produced': company.products_produced_this_step,
                'sold': company.products_sold_this_step,
            })
    
    env._simulate_supply_chain = tracked_simulate
    return transactions

def analyze_parts_company_detailed():
    print("=" * 80)
    print("Parts 公司详细交易追踪")
    print("=" * 80)
    
    # 加载环境
    config = load_config("config/config.json")
    env = IndustryEnv(config.environment)
    env.reset(options={"initial_firms": config.environment.initial_firms})
    
    # 找到一个 Parts 公司
    parts_companies = [i for i, c in enumerate(env.companies) 
                      if sector_relations[c.sector_id].name == "Parts"]
    
    if not parts_companies:
        print("没有找到 Parts 公司")
        return
    
    target_idx = parts_companies[0]
    company = env.companies[target_idx]
    
    print(f"\n目标公司 #{target_idx}:")
    print(f"  初始资本: {company.capital:,.2f}")
    print(f"  采购预算: {company.get_max_purchase_budget():,.2f}")
    print(f"  生产能力: {company.get_max_production():,.2f}")
    print(f"  产品库存: {company.product_inventory:.2f}")
    print(f"  原材料库存: {company.raw_inventory:.2f}")
    print(f"  供应商数量: {len(company.suppliers)}")
    
    # 显示供应商信息
    print(f"\n供应商列表:")
    nearest_5 = sorted(company.suppliers, key=lambda s: company.distance_to(s))[:5]
    for i, sup in enumerate(nearest_5, 1):
        dist = company.distance_to(sup)
        sector = sector_relations[sup.sector_id].name
        print(f"  {i}. {sector} - 距离: {dist:.2f}, 库存: {sup.product_inventory:.2f}, 价格: {sup.revenue_rate:.2f}")
    
    # 修补环境以追踪交易
    transactions = patch_env_to_track_transactions(env, target_idx)
    
    # 模拟5个步骤
    print(f"\n模拟 5 个步骤:")
    print(f"=" * 80)
    
    for step in range(5):
        print(f"\n【Step {step + 1}】")
        
        # 记录步骤前状态
        capital_before = company.capital
        raw_before = company.raw_inventory
        product_before = company.product_inventory
        
        # 执行步骤
        action = np.zeros(env.max_actions_per_step * 3, dtype=np.float32)
        env.step(action)
        
        # 显示变化
        print(f"资本: {capital_before:,.2f} → {company.capital:,.2f} (变化: {company.capital - capital_before:+,.2f})")
        print(f"原材料: {raw_before:.2f} → {company.raw_inventory:.2f} (变化: {company.raw_inventory - raw_before:+,.2f})")
        print(f"产品: {product_before:.2f} → {company.product_inventory:.2f} (变化: {company.product_inventory - product_before:+,.2f})")
        print(f"本步: 购买={company.products_purchased_this_step:.2f}, 生产={company.products_produced_this_step:.2f}, 销售={company.products_sold_this_step:.2f}")
        
        # 检查异常
        if company.raw_inventory - raw_before > 10000:
            print(f"  🔴 警告: 原材料库存爆炸式增长 (+{company.raw_inventory - raw_before:,.2f})！")
            
            # 详细分析：计算理论购买金额
            if company.products_purchased_this_step > 0:
                # 假设Raw价格是1.0
                theoretical_cost = company.products_purchased_this_step * 1.0
                print(f"  理论购买成本: {theoretical_cost:,.2f}")
                print(f"  实际资本变化: {company.capital - capital_before:,.2f}")
                print(f"  采购预算: {company.get_max_purchase_budget():,.2f}")
            
            # 检查供应商状态
            print(f"\n  供应商交易后状态:")
            for i, sup in enumerate(nearest_5[:3], 1):
                sector = sector_relations[sup.sector_id].name
                print(f"    {i}. {sector} - 剩余库存: {sup.product_inventory:.2f}, 本步销售: {sup.products_sold_this_step:.2f}")
        
        if capital_before - company.capital > 50000:
            print(f"  🔴 警告: 资本大幅下降 ({capital_before - company.capital:,.2f})！")
        
        if company.capital < 0:
            print(f"  🔴🔴🔴 严重: 资本变为负数！")
            break

if __name__ == "__main__":
    analyze_parts_company_detailed()

