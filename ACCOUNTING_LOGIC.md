# 供应链会计逻辑完整说明

## 修复后的正确会计逻辑（权责发生制）

### 阶段1：采购阶段（env/env.py 第357-418行）

```python
# 买家从卖家采购
units_to_sell = min(units_requested, supplier.product_inventory)
cost = units_to_sell * unit_price

# 1. 买家获得库存，但不支付货款（记账）
customer.raw_inventory += units_to_sell  # (或parts/elec/battery/oem等)
customer.input_cost_per_unit['raw'] = cost / units_to_sell  # 记录单位成本

# 2. 买家支付运输成本（现金）
if not disable_logistic_costs:
    logistic_cost = rate × price × volume × distance
    customer.capital -= logistic_cost  # ✅ 立即扣现金

# 3. 卖家确认收入和销货成本（记账）
supplier.revenue += units_to_sell × unit_price
supplier.cogs_cost += units_to_sell × supplier.product_unit_cost
supplier.product_inventory -= units_to_sell
```

**关键点**：
- ❌ 买家**不支付货款**（capital不变）
- ✅ 买家**支付运输费**（capital立即扣除）
- ✅ 卖家**确认收入**（revenue累计，capital不变）

---

### 阶段2：生产阶段（env/company.py 第239-346行）

```python
# 使用原材料生产产品
def produce_products():
    # 例如 Parts: 3 Raw → 1 Parts
    craft = min(raw_inventory // 3, production_capacity)
    
    raw_inventory -= 3 × craft
    product_inventory += craft
    
    # 计算产品单位成本（用于将来销售时的COGS）
    raw_cost = input_cost_per_unit['raw']
    product_unit_cost = 3 × raw_cost
```

**关键点**：
- ✅ 消耗原材料库存
- ✅ 生成产品库存
- ✅ 计算产品单位成本（为未来COGS做准备）

---

### 阶段3：销售阶段（env/env.py 第424-436行）

```python
# Service公司销售给市场
if sector_name == "Service" and product_inventory > 0:
    units_sold = product_inventory
    
    # 确认收入（记账）
    revenue += units_sold × revenue_rate
    
    # 确认销货成本（记账）
    cogs_cost += units_sold × product_unit_cost
    
    product_inventory -= units_sold
```

**关键点**：
- ✅ 确认收入（revenue累计，capital不变）
- ✅ 确认COGS（cogs_cost累计，capital不变）
- ✅ 减少库存

---

### 阶段4：结算阶段（env/company.py 第170-198行）

```python
def step(self, max_capital):
    # 1. 计算各项成本
    op_cost = op_cost_rate × capital
    management_cost = capital × 0.001 × (capital / max_capital)^0.5
    
    # 2. 计算利润（revenue和cogs在这里实现）
    total_cost = op_cost + management_cost + cogs_cost
    profit = revenue - total_cost + fixed_income
    
    # 3. 更新资本
    capital += profit
    
    # 4. 清零账户（为下一step准备）
    revenue = 0
    cogs_cost = 0
```

**关键点**：
- ✅ 运营成本和管理成本基于当前capital
- ✅ COGS在此时扣除（之前只是记账）
- ✅ profit加到capital上

---

## 完整的资金流示例

### Example: Service公司一个完整周期

**初始状态**：
- capital: 100,000
- product_inventory: 0
- oem_inventory: 0

**Step 1 - 采购**：
```
采购200个OEM（单价250）
- capital: 100,000（不变）
- oem_inventory: +200
- input_cost_per_unit['oem']: 250
- 运输成本: 100,000 × 0 = 0（假设rate=0）
```

**Step 2 - 生产**：
```
生产100个Service（2 OEM → 1 Service）
- oem_inventory: 200 → 0
- product_inventory: 0 → 100
- product_unit_cost: 2 × 250 = 500
```

**Step 3 - 销售**：
```
卖出100个Service（单价600）
- revenue: +60,000（记账）
- cogs_cost: +50,000（记账，100 × 500）
- product_inventory: 100 → 0
```

**Step 4 - 结算**：
```
op_cost: 100,000 × 0.002 = 200
management_cost: ~100
total_cost: 200 + 100 + 50,000 = 50,300
profit: 60,000 - 50,300 = 9,700
capital: 100,000 + 9,700 = 109,700 ✅
```

---

## 关键修改点

### env/env.py 第363-374行
```python
# 之前（错误）：
customer.capital -= cost  # ❌ 重复扣款

# 现在（正确）：
# 不扣cost，只扣logistics
customer.capital -= logistic_cost  # ✅ 只扣运输费
```

### env/company.py 第189行
```python
# 之前（错误）：
total_cost = op_cost + management_cost + logistic_cost + cogs_cost

# 现在（正确）：
total_cost = op_cost + management_cost + cogs_cost
# logistic_cost已立即支付，不重复扣
```

---

## 验证公式

**每个step的capital变化**：
```
Δcapital = revenue - COGS - op_cost - mgmt_cost - logistic_cost(立即) + fixed_income
```

**其中**：
- revenue: 销售收入
- COGS: 销售产品的生产成本
- op_cost: 运营成本（capital × rate × sector_multiplier）
- mgmt_cost: 管理成本（capital × 0.001 × ...）
- logistic_cost: 运输成本（在采购时立即扣除）
- fixed_income: 固定收入（通常为-5）

---

## 所有tier适用

这套逻辑对所有tier统一适用：

- ✅ Raw → Parts/Electronics/Battery
- ✅ Parts/Electronics/Battery → OEM
- ✅ OEM → Service

所有交易都走同一个采购-生产-销售-结算流程。

