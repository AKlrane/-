"""
Demo showing sector-specific operating cost multipliers.
Demonstrates how different sectors have different cost structures.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env import sector_relations
from env.sector import get_sector_operating_cost_multiplier


def main():
    print("=" * 70)
    print(" " * 15 + "SECTOR OPERATING COST MULTIPLIERS")
    print("=" * 70)
    print()
    print("This demonstrates the sector-specific operating cost structure.")
    print("Higher multipliers mean higher operating costs relative to capital.")
    print()
    print("-" * 70)
    
    # Display all sectors and their cost multipliers
    print(f"{'Sector ID':<12} {'Sector Name':<20} {'Cost Multiplier':<18} {'Impact'}")
    print("-" * 70)
    
    for sector in sector_relations:
        multiplier = sector.operating_cost_multiplier
        
        # Calculate impact description
        if multiplier > 1.0:
            impact = f"+{(multiplier - 1.0) * 100:.0f}% higher costs"
        elif multiplier < 1.0:
            impact = f"{(1.0 - multiplier) * 100:.0f}% lower costs"
        else:
            impact = "Baseline"
        
        print(f"{sector.id:<12} {sector.name:<20} {multiplier:<18.2f} {impact}")
    
    print("-" * 70)
    print()
    
    # Example calculation
    print("=" * 70)
    print("EXAMPLE: Operating Cost Comparison")
    print("=" * 70)
    print()
    print("For a company with $100,000 capital and base op_cost_rate of 5%:")
    print()
    
    base_capital = 100000
    base_rate = 0.05
    
    for sector in sector_relations:
        effective_rate = base_rate * sector.operating_cost_multiplier
        operating_cost = base_capital * effective_rate
        
        print(f"{sector.name:<20} - Effective rate: {effective_rate*100:>5.2f}% - Cost: ${operating_cost:>10,.2f}")
    
    print()
    print("=" * 70)
    print("RATIONALE")
    print("=" * 70)
    print()
    print("Raw Materials (1.5x):")
    print("  • High extraction and processing costs")
    print("  • Expensive mining/refining equipment")
    print("  • Environmental compliance and cleanup")
    print("  • Energy-intensive operations")
    print()
    print("Parts Manufacturing (1.2x):")
    print("  • Manufacturing overhead and maintenance")
    print("  • Quality control and testing")
    print("  • Moderate facility costs")
    print()
    print("Electronics & Battery/Motor (1.1x):")
    print("  • R&D and precision manufacturing")
    print("  • Specialized equipment and materials")
    print("  • Testing and certification")
    print()
    print("OEM/Assembly (1.0x - Baseline):")
    print("  • Primarily assembly operations")
    print("  • Standard manufacturing overhead")
    print()
    print("Service (0.8x):")
    print("  • Lower physical overhead")
    print("  • Less capital-intensive")
    print("  • Primarily labor costs")
    print()
    print("=" * 70)
    print("EXPECTED SIMULATION IMPACT")
    print("=" * 70)
    print()
    print("✓ Raw materials will have higher revenue BUT also higher costs")
    print("✓ More balanced capital distribution across sectors")
    print("✓ Service sector becomes more competitive due to lower costs")
    print("✓ Creates realistic economic trade-offs")
    print("✓ Reflects real-world industry economics")
    print()


if __name__ == "__main__":
    main()
