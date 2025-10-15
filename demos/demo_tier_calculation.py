#!/usr/bin/env python3
"""
Demonstration of dynamic tier calculation from sector relationships.

This script shows how the calculate_sector_tiers() function automatically
determines supply chain tiers based on supplier-consumer relationships.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env import calculate_sector_tiers, sector_relations, SECTOR_TIERS


def main():
    print("=" * 70)
    print("DYNAMIC SECTOR TIER CALCULATION DEMO")
    print("=" * 70)
    
    print("\nSector Relationships:")
    print("-" * 70)
    for sector in sector_relations:
        print(f"Sector {sector.id}: {sector.name}")
        print(f"  Suppliers: {sector.suppliers if sector.suppliers else 'None (produces from scratch)'}")
        print(f"  Consumers: {sector.consumers if sector.consumers else 'None (end consumer)'}")
        print()
    
    print("=" * 70)
    print("CALCULATED TIERS (using calculate_sector_tiers)")
    print("=" * 70)
    
    for sector in sector_relations:
        tier = SECTOR_TIERS[sector.id]
        
        # Determine tier type
        if not sector.suppliers and sector.consumers:
            tier_type = "Root (produces from scratch)"
        elif not sector.suppliers and not sector.consumers:
            tier_type = "Isolated (no supply chain connections)"
        elif not sector.consumers:
            tier_type = "Leaf (end consumer)"
        else:
            tier_type = "Intermediate"
        
        print(f"Sector {sector.id} ({sector.name:15s}): Tier {tier} - {tier_type}")
    
    print("\n" + "=" * 70)
    print("SUPPLY CHAIN FLOW")
    print("=" * 70)
    
    # Group by tier
    tiers_grouped = {}
    for sector in sector_relations:
        tier = SECTOR_TIERS[sector.id]
        if tier not in tiers_grouped:
            tiers_grouped[tier] = []
        tiers_grouped[tier].append(sector.name)
    
    # Display in order
    for tier in sorted(tiers_grouped.keys()):
        sectors_in_tier = ", ".join(tiers_grouped[tier])
        print(f"Tier {tier}: {sectors_in_tier}")
        if tier < max(tiers_grouped.keys()):
            print("   ↓")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("✓ Tier 0: Raw materials (no suppliers)")
    print("✓ Tier 1: Component manufacturers (buy from Raw)")
    print("✓ Tier 2: OEM assembly (buys from components)")
    print("✓ Tier 3: Service providers (buys from OEM)")
    print("✓ Tier 4: Isolated sectors (no connections)")
    print("\nThe system automatically adapts to any supply chain structure!")
    print()


if __name__ == "__main__":
    main()
