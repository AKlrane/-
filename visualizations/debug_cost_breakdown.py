"""
详细追踪公司的成本分解，找出导致死亡的主要原因
"""
import sys
sys.path.append('.')

import numpy as np
from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config

def track_company_costs(env, company, step_num):
    """记录公司在一个步骤中的详细成本"""
    # 记录步骤前状态
    capital_before = company.capital
    revenue_before = company.revenue
    logistic_cost_before = company.logistic_cost
    cogs_cost_before = company.cogs_cost
    
    # 计算运营成本
    op_cost = company.op_cost_rate * company.capital
    
    # 计算管理成本
    max_capital = env.max_capital
    capital_ratio = max(company.capital, 0.0) / max(max_capital, 1.0)
    management_cost = max(company.capital, 0.0) * 0.001 * (capital_ratio ** 0.5)
    
    return {
        'step': step_num,
        'capital_before': capital_before,
        'revenue': revenue_before,
        'op_cost': op_cost,
        'management_cost': management_cost,
        'logistic_cost': logistic_cost_before,
        'cogs_cost': cogs_cost_before,
        'fixed_income': company.fixed_income,
        'purchased': company.products_purchased_this_step,
        'produced': company.products_produced_this_step,
        'sold': company.products_sold_this_step,
        'product_inventory': company.product_inventory,
    }

def analyze_company_death():
    """分析公司死亡的原因"""
    
    print("=" * 80)
    print("公司成本分解与死亡原因分析")
    print("=" * 80)
    
    # 加载环境
    config = load_config("config/config.json")
    env = IndustryEnv(config.environment)
    env.reset(options={"initial_firms": 50})
    
    print(f"\n初始公司数: {len(env.companies)}")
    
    # 找出要追踪的公司：Parts, OEM
    parts_companies = [c for c in env.companies if sector_relations[c.sector_id].name == "Parts"]
    oem_companies = [c for c in env.companies if sector_relations[c.sector_id].name == "OEM"]
    electronics_companies = [c for c in env.companies if sector_relations[c.sector_id].name == "Electronics"]
    battery_companies = [c for c in env.companies if sector_relations[c.sector_id].name == "Battery/Motor"]
    
    # 选择追踪目标
    tracked = {}
    if parts_companies:
        tracked['Parts'] = parts_companies[0]
    if oem_companies:
        tracked['OEM'] = oem_companies[0]
    if electronics_companies:
        tracked['Electronics'] = electronics_companies[0]
    if battery_companies:
        tracked['Battery'] = battery_companies[0]
    
    if not tracked:
        print("❌ 没有找到可追踪的公司")
        return
    
    print(f"\n追踪的公司:")
    for name, company in tracked.items():
        sector = sector_relations[company.sector_id].name
        print(f"  {name}: 初始资本 {company.capital:,.2f}, 位置 ({company.x:.1f}, {company.y:.1f})")
    
    # 记录每个公司的历史
    history = {name: [] for name in tracked.keys()}
    
    # 模拟直到公司死亡或达到最大步数
    max_steps = 100
    action = np.zeros(env.max_actions_per_step * 3, dtype=np.float32)
    
    for step in range(1, max_steps + 1):
        # 在step之前记录当前状态
        step_data = {}
        for name, company in tracked.items():
            if company in env.companies:  # 公司还活着
                step_data[name] = track_company_costs(env, company, step)
        
        # 执行步骤
        obs, reward, done, truncated, info = env.step(action)
        
        # 在step之后记录结果
        for name, company in tracked.items():
            if name in step_data and company in env.companies:
                data = step_data[name]
                data['capital_after'] = company.capital
                data['profit'] = company.capital - data['capital_before']
                data['alive'] = company in env.companies
                history[name].append(data)
            elif name in step_data:
                # 公司在这一步死亡
                data = step_data[name]
                data['capital_after'] = 0
                data['profit'] = -data['capital_before']
                data['alive'] = False
                history[name].append(data)
        
        # 检查所有追踪的公司是否都死了
        all_dead = all(company not in env.companies for company in tracked.values())
        if all_dead:
            print(f"\n所有追踪的公司都在第 {step} 步死亡")
            break
        
        # 每10步报告一次
        if step % 10 == 0:
            alive_count = sum(1 for c in tracked.values() if c in env.companies)
            print(f"Step {step}: {alive_count}/{len(tracked)} 公司存活, 总公司数: {len(env.companies)}")
    
    # 分析结果
    print("\n" + "=" * 80)
    print("成本分解分析")
    print("=" * 80)
    
    for name, data_list in history.items():
        if not data_list:
            continue
        
        print(f"\n【{name}】")
        print(f"  存活步数: {len(data_list)}")
        
        # 计算总计
        total_revenue = sum(d['revenue'] for d in data_list)
        total_op_cost = sum(d['op_cost'] for d in data_list)
        total_management_cost = sum(d['management_cost'] for d in data_list)
        total_logistic_cost = sum(d['logistic_cost'] for d in data_list)
        total_cogs = sum(d['cogs_cost'] for d in data_list)
        total_fixed_income = sum(d['fixed_income'] for d in data_list)
        total_profit = sum(d['profit'] for d in data_list)
        
        total_costs = total_op_cost + total_management_cost + total_logistic_cost + total_cogs
        
        print(f"\n  累计收入与成本:")
        print(f"    收入 (Revenue):        {total_revenue:>12,.2f}")
        print(f"    运营成本 (OpCost):     {total_op_cost:>12,.2f}  ({total_op_cost/total_costs*100:.1f}%)")
        print(f"    管理成本 (MgmtCost):   {total_management_cost:>12,.2f}  ({total_management_cost/total_costs*100:.1f}%)")
        print(f"    物流成本 (Logistics):  {total_logistic_cost:>12,.2f}  ({total_logistic_cost/total_costs*100:.1f}%)")
        print(f"    COGS (成本):           {total_cogs:>12,.2f}  ({total_cogs/total_costs*100:.1f}%)")
        print(f"    固定收入/成本:         {total_fixed_income:>12,.2f}")
        print(f"    {'─' * 50}")
        print(f"    总成本:                {total_costs:>12,.2f}")
        print(f"    净利润:                {total_profit:>12,.2f}")
        
        # 显示最后几步的详细情况
        print(f"\n  最后5步详细数据:")
        print(f"    {'步骤':<6} {'资本':<12} {'收入':<10} {'运营':<10} {'物流':<10} {'COGS':<10} {'利润':<10}")
        for d in data_list[-5:]:
            print(f"    {d['step']:<6} {d['capital_before']:>12,.0f} {d['revenue']:>10,.0f} "
                  f"{d['op_cost']:>10,.0f} {d['logistic_cost']:>10,.0f} "
                  f"{d['cogs_cost']:>10,.0f} {d['profit']:>10,.0f}")
        
        # 分析死亡原因
        if not data_list[-1]['alive']:
            print(f"\n  💀 死亡分析:")
            last_5 = data_list[-5:]
            avg_revenue = np.mean([d['revenue'] for d in last_5])
            avg_costs = np.mean([d['op_cost'] + d['management_cost'] + d['logistic_cost'] + d['cogs_cost'] 
                                for d in last_5])
            avg_profit = np.mean([d['profit'] for d in last_5])
            
            print(f"    最后5步平均收入: {avg_revenue:,.2f}")
            print(f"    最后5步平均成本: {avg_costs:,.2f}")
            print(f"    最后5步平均利润: {avg_profit:,.2f}")
            
            # 找出主要成本
            avg_op = np.mean([d['op_cost'] for d in last_5])
            avg_mgmt = np.mean([d['management_cost'] for d in last_5])
            avg_logistic = np.mean([d['logistic_cost'] for d in last_5])
            avg_cogs = np.mean([d['cogs_cost'] for d in last_5])
            
            costs = [
                ('运营成本', avg_op),
                ('管理成本', avg_mgmt),
                ('物流成本', avg_logistic),
                ('COGS', avg_cogs)
            ]
            costs.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n    主要成本来源（最后5步平均）:")
            for cost_name, cost_value in costs:
                if cost_value > 0:
                    pct = cost_value / avg_costs * 100 if avg_costs > 0 else 0
                    print(f"      {cost_name}: {cost_value:,.2f} ({pct:.1f}%)")
            
            # 检查业务活动
            avg_sold = np.mean([d['sold'] for d in last_5])
            avg_produced = np.mean([d['produced'] for d in last_5])
            print(f"\n    业务活动:")
            print(f"      平均销售: {avg_sold:.2f}")
            print(f"      平均生产: {avg_produced:.2f}")
            
            if avg_sold < 1.0:
                print(f"      🔴 几乎没有销售！需求不足或价格过高")
            if avg_revenue < avg_costs:
                print(f"      🔴 收入低于成本，持续亏损")

if __name__ == "__main__":
    analyze_company_death()

