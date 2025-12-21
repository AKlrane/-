"""
Sector module for industry simulation.
Defines industry sectors and their supply chain relationships.
Based on a 5-tier automotive supply chain model.
"""


class Sector:
    """Represents an industry sector with supply chain relationships."""

    def __init__(
        self,
        id: int,
        name: str,
        suppliers: list[str],
        consumers: list[str],
        operating_cost_multiplier: float = 1.0,
    ):
        self.id = id
        self.name = name
        self.suppliers = suppliers
        self.consumers = consumers
        self.operating_cost_multiplier = operating_cost_multiplier

    def __str__(self):
        return self.name


# Automotive supply chain relationships (5-tier model)
# Based on 5-chain.json configuration
# Operating cost multipliers reflect real-world sector economics:
# - Raw: 1.5x (high extraction, processing, environmental costs)
# - Parts: 1.2x (manufacturing overhead, quality control)
# - Electronics: 1.1x (R&D, precision manufacturing)
# - Battery/Motor: 1.1x (specialized materials, testing)
# - OEM: 1.0x (baseline - assembly focused)
# - Service: 0.8x (lower physical overhead)
# - Other: 1.0x (baseline)
sector_relations = [
    Sector(0, "Raw", [], ["Parts", "Electronics", "Battery/Motor"], operating_cost_multiplier=1.5),
    Sector(1, "Parts", ["Raw"], ["OEM"], operating_cost_multiplier=1.2),
    Sector(2, "Electronics", ["Raw"], ["OEM"], operating_cost_multiplier=1.1),
    Sector(3, "Battery/Motor", ["Raw"], ["OEM"], operating_cost_multiplier=1.1),
    Sector(4, "OEM", ["Parts", "Electronics", "Battery/Motor"], ["Service"], operating_cost_multiplier=0.8),
    Sector(5, "Service", ["OEM"], [], operating_cost_multiplier=0.3),
    Sector(6, "Other", [], [], operating_cost_multiplier=4.0),
]

NUM_SECTORS = 7


def get_sector_operating_cost_multiplier(sector_id: int) -> float:
    """
    Get the operating cost multiplier for a specific sector.
    
    Args:
        sector_id: The ID of the sector
        
    Returns:
        Operating cost multiplier (e.g., 1.5 means 50% higher costs)
    """
    if 0 <= sector_id < len(sector_relations):
        return sector_relations[sector_id].operating_cost_multiplier
    return 1.0  # Default multiplier for unknown sectors


def calculate_sector_tiers(sectors: list[Sector]) -> dict[int, int]:
    """
    Calculate supply chain tiers for all sectors based on supplier relationships.
    
    Tier assignment logic:
    - Tier 0: Sectors with no suppliers but have consumers (most upstream - produce from scratch)
    - Tier N: Sectors that buy from Tier N-1 (one level downstream)
    - Isolated sectors (no suppliers AND no consumers) are assigned after all connected sectors
    
    Uses topological sorting to determine the depth of each sector in the supply chain.
    
    Args:
        sectors: List of Sector objects with supplier/consumer relationships
        
    Returns:
        Dictionary mapping sector_id to tier level
    """
    # Build a name-to-sector mapping for quick lookup
    name_to_sector = {sector.name: sector for sector in sectors}
    
    # Initialize tiers dictionary
    tiers = {}
    
    # Separate isolated sectors (no suppliers AND no consumers)
    isolated_sectors = []
    connected_sectors = []
    
    for sector in sectors:
        if not sector.suppliers and not sector.consumers:
            isolated_sectors.append(sector)
        else:
            connected_sectors.append(sector)
    
    # Find connected sectors with no suppliers (Tier 0 - root of supply chain)
    tier_0_sectors = [s for s in connected_sectors if not s.suppliers]
    for sector in tier_0_sectors:
        tiers[sector.id] = 0
    
    # Iteratively assign tiers based on supplier tiers
    max_iterations = len(connected_sectors)  # Prevent infinite loops
    iteration = 0
    
    while len(tiers) < len(connected_sectors) and iteration < max_iterations:
        iteration += 1
        made_progress = False
        
        for sector in connected_sectors:
            # Skip if already assigned
            if sector.id in tiers:
                continue
            
            # Check if all suppliers have been assigned tiers
            supplier_tiers = []
            all_suppliers_assigned = True
            
            for supplier_name in sector.suppliers:
                if supplier_name in name_to_sector:
                    supplier_sector = name_to_sector[supplier_name]
                    if supplier_sector.id in tiers:
                        supplier_tiers.append(tiers[supplier_sector.id])
                    else:
                        all_suppliers_assigned = False
                        break
            
            # If all suppliers have tiers, assign this sector to max(supplier_tiers) + 1
            if all_suppliers_assigned and supplier_tiers:
                tiers[sector.id] = max(supplier_tiers) + 1
                made_progress = True
            elif all_suppliers_assigned and not supplier_tiers:
                # Sector lists suppliers but they don't exist - treat as Tier 0
                tiers[sector.id] = 0
                made_progress = True
        
        # If we didn't make progress, break (shouldn't happen with valid supply chains)
        if not made_progress:
            break
    
    # Assign isolated sectors to tiers after all connected sectors
    # Each isolated sector gets its own tier
    if isolated_sectors:
        max_tier = max(tiers.values()) if tiers else -1
        for sector in isolated_sectors:
            max_tier += 1
            tiers[sector.id] = max_tier
    
    return tiers


# Calculate supply chain tiers dynamically from sector relationships
# Tier 0: Most upstream (no suppliers - produce from scratch)
# Higher tiers: More downstream (buy from lower tiers)
SECTOR_TIERS = calculate_sector_tiers(sector_relations)
