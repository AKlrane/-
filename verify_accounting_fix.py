"""验证会计修复是否生效"""
import sys
sys.path.append('.')

from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config
import numpy as np
import inspect

print("="*80)
print("验证会计修复是否生效")
print("="*80)

# 检查1：检查env.py中的采购代码
print("\n【检查1】env.py中的采购逻辑:")
env_code = inspect.getsource(IndustryEnv._simulate_supply_chain)
if "customer.capital -= cost" in env_code:
    print("❌ 错误！还在扣采购成本（customer.capital -= cost）")
    print("   修改未生效，可能需要重启Python或清除缓存")
else:
    print("✅ 正确：采购时不扣cost")

if "customer.capital -= logistic_cost" in env_code:
    print("✅ 正确：运输成本立即扣除")
else:
    print("❌ 错误！运输成本没有扣除")

# 检查2：检查company.py中的step()代码
from env.company import Company
print("\n【检查2】company.py中的step()逻辑:")
step_code = inspect.getsource(Company.step)
if "+ self.logistic_cost" in step_code or "self.logistic_cost +" in step_code:
    print("❌ 错误！profit计算中还在加/减logistic_cost")
else:
    print("✅ 正确：profit计算不包含logistic_cost")

# 检查3：实际运行测试
print("\n【检查3】实际运行测试:")
config = load_config("config/config.json")
env = IndustryEnv(config.environment)
env.reset(options={"initial_firms": 50})

# 找一个Service
service = next((c for c in env.companies if sector_relations[c.sector_id].name == "Service"), None)
if not service:
    print("⚠️  没找到Service公司")
else:
    before_capital = service.capital
    
    # 执行供应链
    for c in env.companies:
        c.reset_step_counters()
    env._simulate_supply_chain()
    
    after_supply_chain = service.capital
    capital_change_in_supply_chain = after_supply_chain - before_capital
    
    # 记录财务
    revenue = service.revenue
    cogs = service.cogs_cost
    
    # 手动计算profit
    op_cost = service.op_cost_rate * after_supply_chain
    capital_ratio = after_supply_chain / max(env.max_capital, 1.0)
    mgmt_cost = after_supply_chain * 0.001 * (capital_ratio ** 0.5)
    
    profit = revenue - cogs - op_cost - mgmt_cost + service.fixed_income
    
    # 结算
    service.capital += profit
    service.revenue = 0.0
    service.cogs_cost = 0.0
    service.logistic_cost = 0.0
    
    final_capital = service.capital
    total_change = final_capital - before_capital
    
    print(f"Service测试:")
    print(f"  期初capital: {before_capital:,.0f}")
    print(f"  供应链后: {after_supply_chain:,.0f} (变化: {capital_change_in_supply_chain:+,.0f})")
    print(f"  收入: {revenue:,.0f}")
    print(f"  COGS: {cogs:,.0f}")
    print(f"  营业利润: {profit:,.0f}")
    print(f"  期末capital: {final_capital:,.0f} (总变化: {total_change:+,.0f})")
    
    if capital_change_in_supply_chain < 0:
        print(f"\n  供应链阶段capital减少了{-capital_change_in_supply_chain:,.0f}")
        print(f"  这应该只是运输成本（如果采购了的话）")
    elif capital_change_in_supply_chain == 0:
        print(f"\n  ✅ 供应链阶段capital没变化（正确，因为采购不扣cost）")
    
    if revenue > 0 and cogs > 0:
        gross_margin = (revenue - cogs) / revenue * 100
        print(f"\n  毛利率: {gross_margin:.1f}%")
        if gross_margin < 0:
            print(f"  ❌ 毛利率为负！COGS({cogs:,.0f}) > 收入({revenue:,.0f})")
        elif gross_margin > 0 and gross_margin < 20:
            print(f"  ⚠️  毛利率较低")
        else:
            print(f"  ✅ 毛利率正常")

print("\n" + "="*80)
print("总结")
print("="*80)
print("如果上面显示❌，说明代码修改未生效")
print("请尝试：")
print("1. 删除 __pycache__ 目录: rm -rf env/__pycache__")
print("2. 重新运行: uv run python -m verify_accounting_fix")

