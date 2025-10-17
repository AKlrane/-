"""
Demo: OEM Decoupled Production Logic
Shows how OEM can now produce from any single material type independently.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.company import Company
from env.sector import sector_relations
from config.config import Config


def demo_decoupled_production():
    """Demonstrate OEM decoupled production with different materials."""
    print("=" * 70)
    print("OEM DECOUPLED PRODUCTION DEMO")
    print("=" * 70)
    
    config = Config()
    
    # Find OEM sector ID
    oem_sector_id = None
    for sector in sector_relations:
        if sector.name == "OEM":
            oem_sector_id = sector.id
            break
    
    print("\n🏭 Creating OEM Company...")
    oem = Company(
        capital=500000.0,
        sector_id=oem_sector_id,
        location=(50.0, 50.0),
        tier_prices=config.environment.tier_prices,
        tier_cogs=config.environment.tier_cogs,
        production_capacity_ratio=1.0,  # High capacity
    )
    
    print(f"   Capital: ${oem.capital:,.0f}")
    print(f"   Max Production Capacity: {oem.get_max_production():.0f} units")
    
    # Scenario 1: Produce from parts only
    print("\n" + "─" * 70)
    print("📦 SCENARIO 1: Production from Parts Only")
    print("─" * 70)
    
    oem.parts_inventory = 100.0
    oem.input_cost_per_unit['parts'] = 5.0
    
    print(f"   Parts inventory: {oem.parts_inventory:.0f} units")
    print(f"   Recipe: 20 parts → 1 OEM")
    
    produced = oem.produce_products()
    
    print(f"\n   ✓ Produced: {produced:.0f} OEM units")
    print(f"   ✓ Parts remaining: {oem.parts_inventory:.0f} units")
    print(f"   ✓ Unit cost: ${oem.product_unit_cost:.2f}")
    print(f"   ✓ Total OEM inventory: {oem.product_inventory:.0f} units")
    
    # Scenario 2: Produce from electronics only
    print("\n" + "─" * 70)
    print("⚡ SCENARIO 2: Production from Electronics Only")
    print("─" * 70)
    
    oem.electronics_inventory = 50.0
    oem.input_cost_per_unit['elec'] = 12.0
    
    print(f"   Electronics inventory: {oem.electronics_inventory:.0f} units")
    print(f"   Recipe: 10 electronics → 1 OEM")
    
    produced = oem.produce_products()
    
    print(f"\n   ✓ Produced: {produced:.0f} OEM units")
    print(f"   ✓ Electronics remaining: {oem.electronics_inventory:.0f} units")
    print(f"   ✓ Unit cost: ${oem.product_unit_cost:.2f}")
    print(f"   ✓ Total OEM inventory: {oem.product_inventory:.0f} units")
    
    # Scenario 3: Produce from battery only
    print("\n" + "─" * 70)
    print("🔋 SCENARIO 3: Production from Battery Only")
    print("─" * 70)
    
    oem.battery_inventory = 20.0
    oem.input_cost_per_unit['batt'] = 35.0
    
    print(f"   Battery inventory: {oem.battery_inventory:.0f} units")
    print(f"   Recipe: 4 battery → 1 OEM")
    
    produced = oem.produce_products()
    
    print(f"\n   ✓ Produced: {produced:.0f} OEM units")
    print(f"   ✓ Battery remaining: {oem.battery_inventory:.0f} units")
    print(f"   ✓ Unit cost: ${oem.product_unit_cost:.2f}")
    print(f"   ✓ Total OEM inventory: {oem.product_inventory:.0f} units")
    
    # Scenario 4: Mixed production (all materials available)
    print("\n" + "─" * 70)
    print("🔄 SCENARIO 4: Mixed Production (All Materials)")
    print("─" * 70)
    
    oem.parts_inventory = 60.0
    oem.electronics_inventory = 30.0
    oem.battery_inventory = 12.0
    oem.input_cost_per_unit['parts'] = 5.0
    oem.input_cost_per_unit['elec'] = 12.0
    oem.input_cost_per_unit['batt'] = 35.0
    
    print(f"   Parts inventory: {oem.parts_inventory:.0f} units → can make {int(oem.parts_inventory // 20)} OEM")
    print(f"   Electronics inventory: {oem.electronics_inventory:.0f} units → can make {int(oem.electronics_inventory // 10)} OEM")
    print(f"   Battery inventory: {oem.battery_inventory:.0f} units → can make {int(oem.battery_inventory // 4)} OEM")
    
    produced = oem.produce_products()
    expected = 3 + 3 + 3  # 3 from parts, 3 from elec, 3 from battery
    
    print(f"\n   ✓ Produced: {produced:.0f} OEM units (expected {expected})")
    print(f"   ✓ Parts remaining: {oem.parts_inventory:.0f} units")
    print(f"   ✓ Electronics remaining: {oem.electronics_inventory:.0f} units")
    print(f"   ✓ Battery remaining: {oem.battery_inventory:.0f} units")
    print(f"   ✓ Average unit cost: ${oem.product_unit_cost:.2f}")
    print(f"   ✓ Total OEM inventory: {oem.product_inventory:.0f} units")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print("\n✅ Key Benefits of Decoupled Logic:")
    print("   • Can produce with ANY single material type")
    print("   • No longer blocked by one missing component")
    print("   • Maximizes production capacity utilization")
    print("   • Flexible cost structure based on actual inputs used")
    print("   • Simpler purchasing: buy from nearest K suppliers")
    print("\n📈 Production Flexibility:")
    print("   • 20 parts → 1 OEM")
    print("   • 10 electronics → 1 OEM")
    print("   • 4 battery → 1 OEM")
    print("   • Any combination of the above!")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_decoupled_production()



