"""
Test to verify sector-specific operating costs are applied correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env import IndustryEnv, sector_relations
from config import Config


def test_sector_operating_costs():
    """Test that companies have correct operating cost rates based on their sector."""
    print("=" * 70)
    print("Testing Sector-Specific Operating Costs")
    print("=" * 70)
    print()
    
    # Create environment
    config = Config()
    env = IndustryEnv(config.environment)
    
    # Reset with multiple companies
    obs, _ = env.reset(seed=42, options={"initial_firms": 20})
    
    print(f"Created {len(env.companies)} companies across {len(set(c.sector_id for c in env.companies))} sectors")
    print()
    
    # Group companies by sector
    sector_companies = {}
    for company in env.companies:
        sector_id = company.sector_id
        if sector_id not in sector_companies:
            sector_companies[sector_id] = []
        sector_companies[sector_id].append(company)
    
    # Verify operating cost rates
    base_rate = config.environment.op_cost_rate  # Default from config
    print(f"Base operating cost rate: {base_rate * 100:.2f}%")
    print()
    print("-" * 70)
    print(f"{'Sector':<20} {'Expected':<15} {'Actual':<15} {'Status'}")
    print("-" * 70)
    
    all_correct = True
    
    for sector_id in sorted(sector_companies.keys()):
        sector = sector_relations[sector_id]
        expected_multiplier = sector.operating_cost_multiplier
        expected_rate = base_rate * expected_multiplier
        
        # Check all companies in this sector
        actual_rates = [c.op_cost_rate for c in sector_companies[sector_id]]
        
        # All companies should have the same rate for their sector
        if len(set(actual_rates)) == 1:
            actual_rate = actual_rates[0]
            is_correct = abs(actual_rate - expected_rate) < 1e-10
            status = "✓ PASS" if is_correct else "✗ FAIL"
            
            if not is_correct:
                all_correct = False
                
            print(f"{sector.name:<20} {expected_rate*100:>6.2f}%        {actual_rate*100:>6.2f}%        {status}")
        else:
            all_correct = False
            print(f"{sector.name:<20} {expected_rate*100:>6.2f}%        INCONSISTENT   ✗ FAIL")
            print(f"  Found rates: {[f'{r*100:.2f}%' for r in set(actual_rates)]}")
    
    print("-" * 70)
    print()
    
    if all_correct:
        print("✓ All sectors have correct operating cost multipliers!")
    else:
        print("✗ Some sectors have incorrect operating cost rates!")
    
    print()
    print("=" * 70)
    print("Operating Cost Impact Analysis")
    print("=" * 70)
    print()
    
    # Calculate total operating costs by sector
    print(f"{'Sector':<20} {'Companies':<12} {'Total Capital':<18} {'Total Op Cost'}")
    print("-" * 70)
    
    for sector_id in sorted(sector_companies.keys()):
        sector = sector_relations[sector_id]
        companies = sector_companies[sector_id]
        total_capital = sum(c.capital for c in companies)
        total_op_cost = sum(c.capital * c.op_cost_rate for c in companies)
        
        print(f"{sector.name:<20} {len(companies):<12} ${total_capital:>14,.0f}   ${total_op_cost:>12,.0f}")
    
    print("-" * 70)
    print()
    
    # Show cost differential
    print("=" * 70)
    print("Cost Differential Between Sectors")
    print("=" * 70)
    print()
    
    if 0 in sector_companies and 5 in sector_companies:  # Raw vs Service
        raw_rate = sector_companies[0][0].op_cost_rate
        service_rate = sector_companies[5][0].op_cost_rate
        differential = (raw_rate / service_rate - 1) * 100
        
        print(f"Raw materials operating cost rate: {raw_rate*100:.2f}%")
        print(f"Service sector operating cost rate: {service_rate*100:.2f}%")
        print(f"Differential: Raw costs are {differential:.1f}% higher than Service")
        print()
        print("This means for the same capital:")
        capital_example = 100000
        raw_cost = capital_example * raw_rate
        service_cost = capital_example * service_rate
        print(f"  • Raw company with ${capital_example:,} capital: ${raw_cost:,.2f}/step operating cost")
        print(f"  • Service company with ${capital_example:,} capital: ${service_cost:,.2f}/step operating cost")
        print(f"  • Difference: ${raw_cost - service_cost:,.2f}/step")
    
    print()
    return all_correct


if __name__ == "__main__":
    success = test_sector_operating_costs()
    sys.exit(0 if success else 1)
