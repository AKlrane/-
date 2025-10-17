# OEM 解耦生产逻辑说明

## 概述

OEM公司的生产和购买逻辑已经大幅改造，实现了**解耦**设计。现在OEM不再需要按照固定比例的多种材料才能生产，而是可以使用任意一种材料独立生产。

## 主要变化

### 1. 生产逻辑（Production Logic）

#### 旧逻辑（已弃用）
```
20 parts + 10 electronics + 4 battery → 1 OEM
```
需要**同时**拥有三种材料，按固定比例才能生产。如果某一种材料不足，即使其他材料充足也无法生产。

#### 新逻辑（解耦）
```
20 parts → 1 OEM
OR
10 electronics → 1 OEM  
OR
4 battery/motor → 1 OEM
```

**关键特性：**
- 三种生产路径**完全独立**
- 可以同时使用多种材料生产
- 尽可能多地生产（受产能限制）
- 成本按实际使用的材料计算

#### 示例
```python
# OEM公司库存
parts_inventory = 40         # 可生产 2 个OEM
electronics_inventory = 30   # 可生产 3 个OEM
battery_inventory = 8        # 可生产 2 个OEM

# 生产结果：总共生产 7 个OEM
# - 使用 40 parts 生产 2 个OEM（成本: 2 × 100 = 200）
# - 使用 30 electronics 生产 3 个OEM（成本: 3 × 120 = 360）
# - 使用 8 battery 生产 2 个OEM（成本: 2 × 140 = 280）

# 平均单位成本 = (200 + 360 + 280) / 7 = 120
```

### 2. 购买逻辑（Purchasing Logic）

#### 旧逻辑（已弃用）
- 计算目标库存水平（Parts:Electronics:Battery = 10:5:2）
- 按类型分配预算
- 从各类型的供应商处购买

#### 新逻辑（解耦）
- 找到**最近的K个**上游供应商（默认K=5）
- **不区分类型**，只要是Parts、Electronics或Battery/Motor供应商都可以
- 平均分配预算给这K个供应商
- 购买任何可用的材料

#### 代码实现
```python
# 选择最近的K个供应商
K = 5
nearest_suppliers = sorted(self.suppliers, key=lambda s: self.distance_to(s))[:K]

# 平均分配预算
per_supplier_budget = max_budget / len(nearest_suppliers)

# 从任何Parts/Electronics/Battery供应商购买
for supplier in nearest_suppliers:
    sup_sector = sector_relations[supplier.sector_id].name
    if sup_sector in ("Parts", "Electronics", "Battery/Motor"):
        purchased = self._purchase_from_single_supplier(
            supplier, per_supplier_budget, disable_logistic_costs
        )
```

## 优势

### 1. 灵活性 ✓
- OEM不再被单一材料短缺卡住
- 可以根据市场供应情况灵活调整

### 2. 简化复杂度 ✓
- 不需要复杂的库存平衡逻辑
- 购买决策更简单（距离优先）

### 3. 更真实 ✓
- 模拟现实中的替代材料方案
- 体现供应链的多样性

### 4. 提高产能利用率 ✓
- 即使只有一种材料也能生产
- 减少闲置库存

## 生产配方

| 输入材料 | 数量 | 输出 | 成本计算 |
|---------|------|------|---------|
| Parts | 20 | 1 OEM | 20 × parts_unit_cost |
| Electronics | 10 | 1 OEM | 10 × elec_unit_cost |
| Battery/Motor | 4 | 1 OEM | 4 × batt_unit_cost |

## 测试验证

运行测试以验证新逻辑：
```bash
uv run python tests/test_oem_decoupled.py
```

测试覆盖：
- ✓ 仅使用parts生产
- ✓ 仅使用electronics生产
- ✓ 仅使用battery生产
- ✓ 混合使用多种材料生产
- ✓ 从最近K个供应商购买（不分类型）

## 相关文件

- `env/company.py`: 
  - `produce_products()` - 生产逻辑（第327-364行）
  - `purchase_from_suppliers()` - 购买逻辑（第437-450行）
  - `calculate_oem_needs()` - 简化（第260-265行）
  
- `tests/test_oem_decoupled.py`: 完整的单元测试

## 向后兼容性

此改动**不向后兼容**旧的OEM逻辑。如需旧逻辑，请查看git历史记录。

## 未来改进

可能的优化方向：
1. 动态调整K值（供应商数量）
2. 根据材料价格智能选择生产路径
3. 添加材料偏好权重
4. 考虑运输成本对购买决策的影响

