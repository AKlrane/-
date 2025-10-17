"""
详细检查交易逻辑的每个环节
"""
import sys
sys.path.append('.')

import numpy as np
from env.env import IndustryEnv
from env.sector import sector_relations
from config.config import load_config

output = open("transaction_check.txt", "w", encoding="utf-8")

def log(msg):
    print(msg)
    output.write(msg + "\n")
    output.flush()

log("="*80)
log("交易逻辑详细检查")
log("="*80)

# 加载环境
config = load_config("config/config.json")
env = IndustryEnv(config.environment)
env.reset(options={"initial_firms": 30})

log(f"\n初始公司数: {len(env.companies)}")

# 找一个Parts公司和它的Raw供应商
parts = next((c for c in env.companies if sector_relations[c.sector_id].name == "Parts"), None)

if not parts:
    log("❌ 没找到Parts公司")
    output.close()
    sys.exit(1)

log(f"\n【Parts公司】")
log(f"  位置: ({parts.x:.1f}, {parts.y:.1f})")
log(f"  初始资本: {parts.capital:,.2f}")
log(f"  售价 (revenue_rate): {parts.revenue_rate}")
log(f"  采购预算比率: {parts.purchase_budget_ratio}")
log(f"  采购预算: {parts.get_max_purchase_budget():,.2f}")
log(f"  供应商数量: {len(parts.suppliers)}")

# 检查供应商
raw_suppliers = [s for s in parts.suppliers if sector_relations[s.sector_id].name == "Raw"]
log(f"  其中Raw供应商: {len(raw_suppliers)}")

if not raw_suppliers:
    log("❌ Parts公司没有Raw供应商")
    output.close()
    sys.exit(1)

# 按距离排序，找最近的3个
nearest_raw = sorted(raw_suppliers, key=lambda s: parts.distance_to(s))[:3]
log(f"\n  最近的3个Raw供应商:")
for i, raw in enumerate(nearest_raw, 1):
    dist = parts.distance_to(raw)
    log(f"    {i}. 距离={dist:.2f}, 资本={raw.capital:,.0f}, 库存={raw.product_inventory:.2f}, 售价={raw.revenue_rate}")

# 现在我们手动模拟一次交易，记录每个步骤

log("\n" + "="*80)
log("手动模拟交易流程")
log("="*80)

# ===== 步骤0: 记录初始状态 =====
log("\n【步骤0】记录初始状态")
log(f"  Parts资本: {parts.capital:,.2f}")
log(f"  Parts原材料库存: {parts.raw_inventory:.2f}")

initial_parts_capital = parts.capital
initial_parts_raw = parts.raw_inventory

for i, raw in enumerate(nearest_raw, 1):
    log(f"  Raw{i}资本: {raw.capital:,.2f}, 库存: {raw.product_inventory:.2f}")

# ===== 步骤1: Parts计算采购预算和需求 =====
log("\n【步骤1】Parts计算采购预算")
max_budget = parts.get_max_purchase_budget()
log(f"  最大采购预算 = 资本 × {parts.purchase_budget_ratio} = {parts.capital:,.2f} × {parts.purchase_budget_ratio} = {max_budget:,.2f}")

# 选择最近K个供应商
K = 5
nearest_k = sorted(parts.suppliers, key=lambda s: parts.distance_to(s))[:K]
log(f"  选择最近{K}个供应商（实际有{len(nearest_k)}个）")

# Parts会平均分配预算给每个供应商
per_supplier_budget = max_budget / len(nearest_k)
log(f"  每个供应商的预算 = {max_budget:,.2f} / {len(nearest_k)} = {per_supplier_budget:,.2f}")

# ===== 步骤2: 收集订单 =====
log("\n【步骤2】Parts向供应商下订单")
orders_to_raw = []
for supplier in nearest_k:
    sup_sector = sector_relations[supplier.sector_id].name
    if sup_sector == "Raw":
        unit_price = supplier.revenue_rate
        units_requested = per_supplier_budget / max(unit_price, 1e-8)
        dist = parts.distance_to(supplier)
        orders_to_raw.append({
            'supplier': supplier,
            'distance': dist,
            'units_requested': units_requested,
            'unit_price': unit_price
        })
        log(f"  → Raw供应商 (距离={dist:.2f})")
        log(f"     预算={per_supplier_budget:,.2f}, 单价={unit_price}, 请求数量={units_requested:,.2f}")

# ===== 步骤3: 供应商按距离排序订单 =====
log("\n【步骤3】Raw供应商处理订单（从近到远）")
# 对于每个Raw供应商
for raw in nearest_raw[:1]:  # 只检查第一个最近的
    # 找到发给这个供应商的订单
    orders_to_this_raw = [o for o in orders_to_raw if o['supplier'] == raw]
    if not orders_to_this_raw:
        continue
    
    order = orders_to_this_raw[0]
    log(f"\n  Raw供应商 (距离={order['distance']:.2f}):")
    log(f"    当前库存: {raw.product_inventory:.2f}")
    log(f"    收到订单: {order['units_requested']:.2f} 单位")
    
    # 计算可以卖多少
    units_requested = order['units_requested']
    supplier_inventory = raw.product_inventory
    customer_capital = parts.capital
    unit_price = order['unit_price']
    
    # 客户最多能买多少
    max_affordable_units = customer_capital / max(unit_price, 1e-8)
    
    # 实际成交量
    units_to_sell = min(units_requested, supplier_inventory, max_affordable_units)
    
    log(f"    限制条件:")
    log(f"      请求数量: {units_requested:.2f}")
    log(f"      供应商库存: {supplier_inventory:.2f}")
    log(f"      客户支付能力: {max_affordable_units:.2f}")
    log(f"    → 实际成交: {units_to_sell:.2f} 单位")
    
    # ===== 步骤4: 执行交易 =====
    log(f"\n【步骤4】执行交易")
    
    cost = units_to_sell * unit_price
    log(f"  货款 = {units_to_sell:.2f} × {unit_price} = {cost:,.2f}")
    
    # 计算物流成本
    if not env.disable_logistic_costs:
        dist = order['distance']
        logistic_cost = raw.logistic_cost_rate * unit_price * units_to_sell * max(dist, raw.min_distance_epsilon)
        log(f"  物流成本 = {raw.logistic_cost_rate} × {unit_price} × {units_to_sell:.2f} × {dist:.2f} = {logistic_cost:,.2f}")
    else:
        logistic_cost = 0
        log(f"  物流成本: 已禁用")
    
    total_cost_to_buyer = cost + logistic_cost
    log(f"  买家总支出 = {cost:,.2f} + {logistic_cost:,.2f} = {total_cost_to_buyer:,.2f}")
    
    # 卖家收入
    # 注意: add_revenue会乘以revenue_rate
    revenue_to_seller = raw.revenue_rate * units_to_sell
    log(f"  卖家收入 = revenue_rate × 数量 = {raw.revenue_rate} × {units_to_sell:.2f} = {revenue_to_seller:,.2f}")
    
    log(f"\n  ⚠️ 注意: 卖家收入计算中revenue_rate被乘了两次！")
    log(f"     第一次: cost = units_to_sell × unit_price (unit_price=revenue_rate)")
    log(f"     第二次: add_revenue方法中 revenue += revenue_rate × units_to_sell")
    log(f"     实际应该: 卖家收入 = cost = {cost:,.2f}")

# ===== 执行真实的env.step =====
log("\n" + "="*80)
log("执行真实的 env.step()")
log("="*80)

# Monkey-patch _simulate_supply_chain 来拦截交易
original_simulate = env._simulate_supply_chain
transaction_log = []

def tracked_simulate():
    # 在供应链模拟前，标记一个Parts公司
    if parts in env.companies:
        parts._tracked_capital_before = parts.capital
        parts._tracked_raw_before = parts.raw_inventory
    
    # 在Raw供应商上也做标记
    for raw in nearest_raw[:1]:
        if raw in env.companies:
            raw._tracked_capital_before = raw.capital
            raw._tracked_inventory_before = raw.product_inventory
            raw._tracked_revenue_before = raw.revenue
    
    # 调用原始方法
    original_simulate()
    
    # 记录交易后的状态（在step()调用前）
    if parts in env.companies:
        transaction_log.append({
            'parts_capital_after_trade': parts.capital,
            'parts_raw_after_trade': parts.raw_inventory,
            'parts_revenue_accumulated': parts.revenue,
            'parts_logistic_accumulated': parts.logistic_cost,
            'parts_cogs_accumulated': parts.cogs_cost,
        })
    
    for raw in nearest_raw[:1]:
        if raw in env.companies:
            transaction_log.append({
                'raw_capital_after_trade': raw.capital,
                'raw_inventory_after_trade': raw.product_inventory,
                'raw_revenue_accumulated': raw.revenue,
                'raw_sold': raw.products_sold_this_step,
            })

env._simulate_supply_chain = tracked_simulate

# 执行一步
action = np.zeros(env.max_actions_per_step * 3, dtype=np.float32)
env.step(action)

log(f"\n【交易后状态】（在各公司的step()方法调用后）")
log(f"  Parts:")
log(f"    资本: {initial_parts_capital:,.2f} → {parts.capital:,.2f} (变化: {parts.capital - initial_parts_capital:+,.2f})")
log(f"    原材料库存: {initial_parts_raw:.2f} → {parts.raw_inventory:.2f} (变化: {parts.raw_inventory - initial_parts_raw:+.2f})")
log(f"    本步购买: {parts.products_purchased_this_step:.2f}")

for i, raw in enumerate(nearest_raw[:1], 1):
    if raw in env.companies and hasattr(raw, '_tracked_capital_before'):
        log(f"  Raw{i}:")
        log(f"    资本: {raw._tracked_capital_before:,.2f} → {raw.capital:,.2f} (变化: {raw.capital - raw._tracked_capital_before:+,.2f})")
        log(f"    库存: {raw._tracked_inventory_before:.2f} → {raw.product_inventory:.2f} (变化: {raw.product_inventory - raw._tracked_inventory_before:+.2f})")
        log(f"    本步销售: {raw.products_sold_this_step:.2f}")

# 检查问题
log("\n" + "="*80)
log("问题诊断")
log("="*80)

if transaction_log:
    for i, entry in enumerate(transaction_log):
        log(f"\n交易记录 #{i+1}:")
        for key, value in entry.items():
            if isinstance(value, float):
                log(f"  {key}: {value:,.2f}")
            else:
                log(f"  {key}: {value}")

# 检查Parts是否真的购买了东西
if parts.raw_inventory - initial_parts_raw > 0:
    log(f"\n✅ Parts成功购买了 {parts.raw_inventory - initial_parts_raw:.2f} 单位原材料")
    
    # 检查资本变化是否合理
    capital_decrease = initial_parts_capital - parts.capital
    log(f"   Parts资本减少: {capital_decrease:,.2f}")
    
    # 理论上应该 = 货款 + 物流成本 + 运营成本
    raw_price = nearest_raw[0].revenue_rate
    purchased_amount = parts.raw_inventory - initial_parts_raw
    theoretical_cost = purchased_amount * raw_price
    log(f"   理论货款: {purchased_amount:.2f} × {raw_price} = {theoretical_cost:,.2f}")
    
    if abs(capital_decrease - theoretical_cost) > theoretical_cost * 0.5:  # 差异超过50%
        log(f"   ⚠️ 资本变化与理论货款差异较大！")
else:
    log(f"\n❌ Parts没有购买到任何原材料")
    log(f"   可能原因:")
    log(f"     1. 采购预算不足")
    log(f"     2. 供应商库存不足")
    log(f"     3. 采购逻辑有问题")

# 检查Raw是否收到钱
for raw in nearest_raw[:1]:
    if hasattr(raw, '_tracked_capital_before'):
        capital_increase = raw.capital - raw._tracked_capital_before
        if capital_increase > 0:
            log(f"\n✅ Raw供应商资本增加了 {capital_increase:,.2f}")
        elif capital_increase < 0:
            log(f"\n⚠️ Raw供应商资本反而减少了 {capital_increase:,.2f}")
            log(f"   可能是运营成本超过了销售收入")
        else:
            log(f"\n❌ Raw供应商资本没有变化")

log("\n" + "="*80)
log("检查完成")
log("="*80)
output.close()
print("\n✅ 结果已保存到 transaction_check.txt")

