"""
Test script for new decoupled OEM production logic.
Tests:
1. OEM can produce from 20 parts alone
2. OEM can produce from 10 electronics alone
3. OEM can produce from 4 battery alone
4. OEM purchases from nearest K suppliers regardless of type
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.company import Company
from env.sector import sector_relations
from config.config import Config

def test_oem_production_from_parts():
    """Test OEM production using only parts: 20 parts -> 1 OEM"""
    print("\n=== Test 1: OEM Production from Parts Only ===")
    
    config = Config()
    
    # Find OEM sector ID
    oem_sector_id = None
    for sector in sector_relations:
        if sector.name == "OEM":
            oem_sector_id = sector.id
            break
    
    # Create OEM company
    oem = Company(
        capital=100000.0,
        sector_id=oem_sector_id,
        location=(50.0, 50.0),
        tier_prices=config.environment.tier_prices,
        tier_cogs=config.environment.tier_cogs,
        production_capacity_ratio=0.5,
    )
    
    # Manually add parts inventory
    oem.parts_inventory = 60.0  # Should be able to produce 3 OEM (60 / 20)
    oem.input_cost_per_unit['parts'] = 5.0
    
    print(f"Initial parts inventory: {oem.parts_inventory}")
    print(f"Initial product inventory: {oem.product_inventory}")
    
    # Produce
    produced = oem.produce_products()
    
    print(f"Produced: {produced} OEM units")
    print(f"Remaining parts inventory: {oem.parts_inventory}")
    print(f"Final product inventory: {oem.product_inventory}")
    print(f"Product unit cost: {oem.product_unit_cost}")
    
    # Should produce 3 OEM and consume 60 parts
    assert produced == 3.0, f"Expected 3 OEM, got {produced}"
    assert oem.parts_inventory == 0.0, f"Expected 0 parts left, got {oem.parts_inventory}"
    assert oem.product_unit_cost == 100.0, f"Expected cost 100 (20*5), got {oem.product_unit_cost}"
    print("✓ Test passed!")


def test_oem_production_from_electronics():
    """Test OEM production using only electronics: 10 electronics -> 1 OEM"""
    print("\n=== Test 2: OEM Production from Electronics Only ===")
    
    config = Config()
    oem_sector_id = None
    for sector in sector_relations:
        if sector.name == "OEM":
            oem_sector_id = sector.id
            break
    
    oem = Company(
        capital=100000.0,
        sector_id=oem_sector_id,
        location=(50.0, 50.0),
        tier_prices=config.environment.tier_prices,
        tier_cogs=config.environment.tier_cogs,
        production_capacity_ratio=0.5,
    )
    
    # Manually add electronics inventory
    oem.electronics_inventory = 25.0  # Should be able to produce 2 OEM (25 / 10)
    oem.input_cost_per_unit['elec'] = 12.0
    
    print(f"Initial electronics inventory: {oem.electronics_inventory}")
    print(f"Initial product inventory: {oem.product_inventory}")
    
    # Produce
    produced = oem.produce_products()
    
    print(f"Produced: {produced} OEM units")
    print(f"Remaining electronics inventory: {oem.electronics_inventory}")
    print(f"Final product inventory: {oem.product_inventory}")
    print(f"Product unit cost: {oem.product_unit_cost}")
    
    # Should produce 2 OEM and consume 20 electronics
    assert produced == 2.0, f"Expected 2 OEM, got {produced}"
    assert oem.electronics_inventory == 5.0, f"Expected 5 electronics left, got {oem.electronics_inventory}"
    assert oem.product_unit_cost == 120.0, f"Expected cost 120 (10*12), got {oem.product_unit_cost}"
    print("✓ Test passed!")


def test_oem_production_from_battery():
    """Test OEM production using only battery: 4 battery -> 1 OEM"""
    print("\n=== Test 3: OEM Production from Battery Only ===")
    
    config = Config()
    oem_sector_id = None
    for sector in sector_relations:
        if sector.name == "OEM":
            oem_sector_id = sector.id
            break
    
    oem = Company(
        capital=100000.0,
        sector_id=oem_sector_id,
        location=(50.0, 50.0),
        tier_prices=config.environment.tier_prices,
        tier_cogs=config.environment.tier_cogs,
        production_capacity_ratio=0.5,
    )
    
    # Manually add battery inventory
    oem.battery_inventory = 10.0  # Should be able to produce 2 OEM (10 / 4)
    oem.input_cost_per_unit['batt'] = 35.0
    
    print(f"Initial battery inventory: {oem.battery_inventory}")
    print(f"Initial product inventory: {oem.product_inventory}")
    
    # Produce
    produced = oem.produce_products()
    
    print(f"Produced: {produced} OEM units")
    print(f"Remaining battery inventory: {oem.battery_inventory}")
    print(f"Final product inventory: {oem.product_inventory}")
    print(f"Product unit cost: {oem.product_unit_cost}")
    
    # Should produce 2 OEM and consume 8 battery
    assert produced == 2.0, f"Expected 2 OEM, got {produced}"
    assert oem.battery_inventory == 2.0, f"Expected 2 battery left, got {oem.battery_inventory}"
    assert oem.product_unit_cost == 140.0, f"Expected cost 140 (4*35), got {oem.product_unit_cost}"
    print("✓ Test passed!")


def test_oem_production_mixed():
    """Test OEM production with mixed inventory - should produce from all types"""
    print("\n=== Test 4: OEM Production with Mixed Inventory ===")
    
    config = Config()
    oem_sector_id = None
    for sector in sector_relations:
        if sector.name == "OEM":
            oem_sector_id = sector.id
            break
    
    oem = Company(
        capital=100000.0,
        sector_id=oem_sector_id,
        location=(50.0, 50.0),
        tier_prices=config.environment.tier_prices,
        tier_cogs=config.environment.tier_cogs,
        production_capacity_ratio=1.0,  # High capacity
    )
    
    # Add mixed inventory
    oem.parts_inventory = 40.0  # Can produce 2 OEM
    oem.electronics_inventory = 30.0  # Can produce 3 OEM
    oem.battery_inventory = 8.0  # Can produce 2 OEM
    oem.input_cost_per_unit['parts'] = 5.0
    oem.input_cost_per_unit['elec'] = 12.0
    oem.input_cost_per_unit['batt'] = 35.0
    
    print(f"Initial inventory - Parts: {oem.parts_inventory}, Electronics: {oem.electronics_inventory}, Battery: {oem.battery_inventory}")
    
    # Produce
    produced = oem.produce_products()
    
    print(f"Produced: {produced} OEM units (should be 2+3+2=7)")
    print(f"Remaining inventory - Parts: {oem.parts_inventory}, Electronics: {oem.electronics_inventory}, Battery: {oem.battery_inventory}")
    print(f"Product unit cost: {oem.product_unit_cost}")
    
    # Should produce 7 OEM total (2 from parts, 3 from electronics, 2 from battery)
    assert produced == 7.0, f"Expected 7 OEM, got {produced}"
    assert oem.parts_inventory == 0.0, "Expected 0 parts left"
    assert oem.electronics_inventory == 0.0, "Expected 0 electronics left"
    assert oem.battery_inventory == 0.0, "Expected 0 battery left"
    
    # Average cost should be (2*100 + 3*120 + 2*140) / 7 = (200 + 360 + 280) / 7 = 120
    expected_cost = (2*100.0 + 3*120.0 + 2*140.0) / 7.0
    assert abs(oem.product_unit_cost - expected_cost) < 0.01, f"Expected cost {expected_cost}, got {oem.product_unit_cost}"
    print("✓ Test passed!")


def test_oem_purchase_decoupled():
    """Test OEM purchasing from nearest K suppliers regardless of type"""
    print("\n=== Test 5: OEM Decoupled Purchasing ===")
    
    config = Config()
    
    # Find sector IDs
    oem_sector_id = None
    parts_sector_id = None
    elec_sector_id = None
    batt_sector_id = None
    
    for sector in sector_relations:
        if sector.name == "OEM":
            oem_sector_id = sector.id
        elif sector.name == "Parts":
            parts_sector_id = sector.id
        elif sector.name == "Electronics":
            elec_sector_id = sector.id
        elif sector.name == "Battery/Motor":
            batt_sector_id = sector.id
    
    # Create OEM company
    oem = Company(
        capital=100000.0,
        sector_id=oem_sector_id,
        location=(50.0, 50.0),
        tier_prices=config.environment.tier_prices,
        tier_cogs=config.environment.tier_cogs,
        purchase_budget_ratio=0.5,
    )
    
    # Create supplier companies at different distances
    parts_supplier_near = Company(
        capital=50000.0,
        sector_id=parts_sector_id,
        location=(51.0, 51.0),  # Very close
        tier_prices=config.environment.tier_prices,
        tier_cogs=config.environment.tier_cogs,
    )
    parts_supplier_near.product_inventory = 100.0
    
    elec_supplier_near = Company(
        capital=50000.0,
        sector_id=elec_sector_id,
        location=(52.0, 50.0),  # Close
        tier_prices=config.environment.tier_prices,
        tier_cogs=config.environment.tier_cogs,
    )
    elec_supplier_near.product_inventory = 100.0
    
    batt_supplier_far = Company(
        capital=50000.0,
        sector_id=batt_sector_id,
        location=(80.0, 80.0),  # Far
        tier_prices=config.environment.tier_prices,
        tier_cogs=config.environment.tier_cogs,
    )
    batt_supplier_far.product_inventory = 100.0
    
    # Add suppliers
    oem.suppliers = [parts_supplier_near, elec_supplier_near, batt_supplier_far]
    
    print(f"OEM initial capital: {oem.capital}")
    print(f"OEM purchase budget: {oem.get_max_purchase_budget()}")
    print(f"Suppliers: Parts (dist={oem.distance_to(parts_supplier_near):.2f}), " +
          f"Electronics (dist={oem.distance_to(elec_supplier_near):.2f}), " +
          f"Battery (dist={oem.distance_to(batt_supplier_far):.2f})")
    
    # Purchase
    initial_parts = oem.parts_inventory
    initial_elec = oem.electronics_inventory
    initial_batt = oem.battery_inventory
    
    purchased = oem.purchase_from_suppliers(disable_logistic_costs=True)
    
    print(f"\nTotal purchased: {purchased}")
    print(f"Parts inventory: {initial_parts} -> {oem.parts_inventory}")
    print(f"Electronics inventory: {initial_elec} -> {oem.electronics_inventory}")
    print(f"Battery inventory: {initial_batt} -> {oem.battery_inventory}")
    
    # Should purchase from all three types (nearest K=5 includes all 3)
    assert oem.parts_inventory > initial_parts, "Should have purchased parts"
    assert oem.electronics_inventory > initial_elec, "Should have purchased electronics"
    assert oem.battery_inventory > initial_batt, "Should have purchased battery"
    assert purchased > 0, "Should have purchased something"
    
    print("✓ Test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing New Decoupled OEM Logic")
    print("=" * 60)
    
    try:
        test_oem_production_from_parts()
        test_oem_production_from_electronics()
        test_oem_production_from_battery()
        test_oem_production_mixed()
        test_oem_purchase_decoupled()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

