"""输出诊断结果到文件"""
import sys
sys.path.append('.')

import numpy as np
from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config

output = open("diagnosis_result.txt", "w", encoding="utf-8")

def log(msg):
    print(msg)
    output.write(msg + "\n")
    output.flush()

log("="*80)
log("成本诊断分析")
log("="*80)

config = load_config("config/config.json")
env = IndustryEnv(config.environment)
env.reset(options={"initial_firms": 50})

log(f"\n初始公司数: {len(env.companies)}")

# 找目标公司
parts = next((c for c in env.companies if sector_relations[c.sector_id].name == "Parts"), None)
oem = next((c for c in env.companies if sector_relations[c.sector_id].name == "OEM"), None)
raw = next((c for c in env.companies if sector_relations[c.sector_id].name == "Raw"), None)

if parts:
    log(f"\n【Parts公司】")
    log(f"  初始资本: {parts.capital:,.2f}")
    log(f"  运营成本率: {parts.op_cost_rate}")
    log(f"  售价: {parts.revenue_rate}")
    log(f"  产品单位成本: {parts.product_unit_cost}")
    log(f"  采购预算(10%): {parts.get_max_purchase_budget():,.2f}")
    log(f"  生产能力(10%): {parts.get_max_production():,.2f}")

if oem:
    log(f"\n【OEM公司】")
    log(f"  初始资本: {oem.capital:,.2f}")
    log(f"  运营成本率: {oem.op_cost_rate}")
    log(f"  售价: {oem.revenue_rate}")
    log(f"  产品单位成本: {oem.product_unit_cost}")
    log(f"  采购预算(10%): {oem.get_max_purchase_budget():,.2f}")
    log(f"  生产能力(10%): {oem.get_max_production():,.2f}")

if raw:
    log(f"\n【Raw公司（对照）】")
    log(f"  初始资本: {raw.capital:,.2f}")
    log(f"  运营成本率: {raw.op_cost_rate}")
    log(f"  售价: {raw.revenue_rate}")
    log(f"  产品单位成本: {raw.product_unit_cost}")

# 模拟20步
action = np.zeros(env.max_actions_per_step * 3, dtype=np.float32)

log("\n" + "="*80)
log("模拟20步详细数据")
log("="*80)

for step in range(1, 21):
    # 记录前状态
    states = {}
    for name, company in [("Parts", parts), ("OEM", oem), ("Raw", raw)]:
        if company and company in env.companies:
            states[name] = {
                'capital': company.capital,
                'revenue': company.revenue,
                'logistic': company.logistic_cost,
                'cogs': company.cogs_cost,
                'op_rate': company.op_cost_rate,
            }
    
    # 执行
    env.step(action)
    
    log(f"\n【Step {step}】总公司数: {len(env.companies)}")
    
    for name, company in [("Parts", parts), ("OEM", oem), ("Raw", raw)]:
        if company and name in states:
            if company in env.companies:
                s = states[name]
                profit = company.capital - s['capital']
                op_cost = s['op_rate'] * s['capital']
                
                # 计算管理成本
                capital_ratio = max(s['capital'], 0.0) / max(env.max_capital, 1.0)
                mgmt_cost = max(s['capital'], 0.0) * 0.001 * (capital_ratio ** 0.5)
                
                total_cost = op_cost + mgmt_cost + s['logistic'] + s['cogs']
                
                log(f"  {name}:")
                log(f"    资本: {s['capital']:>10,.0f} → {company.capital:>10,.0f} (利润: {profit:+,.0f})")
                log(f"    收入: {s['revenue']:>10,.0f}")
                log(f"    成本明细:")
                log(f"      运营成本: {op_cost:>10,.0f}  ({op_cost/total_cost*100:.1f}%)")
                log(f"      管理成本: {mgmt_cost:>10,.0f}  ({mgmt_cost/total_cost*100:.1f}%)")
                log(f"      物流成本: {s['logistic']:>10,.0f}  ({s['logistic']/total_cost*100:.1f}%)")
                log(f"      COGS:     {s['cogs']:>10,.0f}  ({s['cogs']/total_cost*100:.1f}%)")
                log(f"      总成本:   {total_cost:>10,.0f}")
                log(f"    业务: 销售={company.products_sold_this_step:.0f}, 生产={company.products_produced_this_step:.0f}")
                
                # 预警
                if s['revenue'] == 0:
                    log(f"    🔴 警告: 没有收入！")
                if s['revenue'] < total_cost:
                    deficit = total_cost - s['revenue']
                    log(f"    🔴 警告: 亏损 {deficit:,.0f} (收入不足以覆盖成本)")
                if company.products_sold_this_step == 0:
                    log(f"    🔴 警告: 没有销售！")
            else:
                log(f"  {name}: 💀 已死亡")

log("\n" + "="*80)
log("诊断完成")
log("="*80)
output.close()

print("\n✅ 结果已保存到 diagnosis_result.txt")

