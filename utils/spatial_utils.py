"""
Spatial utilities for the Industry Simulation.
Provides helper functions for working with company locations and spatial relationships.
"""

import numpy as np
from typing import List, Tuple
from env import Company


def find_nearest_companies(company: Company, all_companies: List[Company], n: int = 5) -> List[Tuple[Company, float]]:
    """
    Find the n nearest companies to a given company.
    
    Args:
        company: The reference company
        all_companies: List of all companies to search
        n: Number of nearest neighbors to return
        
    Returns:
        List of (company, distance) tuples, sorted by distance
    """
    distances = []
    for other in all_companies:
        if other is not company:  # Don't include the company itself
            dist = company.distance_to(other)
            distances.append((other, dist))
    
    # Sort by distance and return top n
    distances.sort(key=lambda x: x[1])
    return distances[:n]


def find_companies_in_radius(
    center: Tuple[float, float], 
    all_companies: List[Company], 
    radius: float
) -> List[Company]:
    """
    Find all companies within a certain radius of a point.
    
    Args:
        center: (x, y) coordinates of the center point
        all_companies: List of all companies
        radius: Search radius
        
    Returns:
        List of companies within the radius
    """
    cx, cy = center
    companies_in_radius = []
    
    for company in all_companies:
        distance = np.sqrt((company.x - cx)**2 + (company.y - cy)**2)
        if distance <= radius:
            companies_in_radius.append(company)
    
    return companies_in_radius


def calculate_sector_clustering(all_companies: List[Company], sector_id: int) -> float:
    """
    Measure how clustered companies in a sector are.
    Returns average distance to nearest same-sector neighbor.
    
    Args:
        all_companies: List of all companies
        sector_id: The sector to analyze
        
    Returns:
        Average distance to nearest same-sector neighbor
    """
    sector_companies = [c for c in all_companies if c.sector_id == sector_id]
    
    if len(sector_companies) < 2:
        return float('inf')
    
    total_distance = 0.0
    
    for company in sector_companies:
        # Find nearest company in same sector
        min_distance = float('inf')
        for other in sector_companies:
            if other is not company:
                dist = company.distance_to(other)
                min_distance = min(min_distance, dist)
        total_distance += min_distance
    
    return total_distance / len(sector_companies)


def suggest_optimal_location(
    all_companies: List[Company],
    sector_id: int,
    size: float,
    strategy: str = "cluster"
) -> Tuple[float, float]:
    """
    Suggest an optimal location for a new company based on strategy.
    
    Args:
        all_companies: List of existing companies
        sector_id: Sector of the new company
        size: Size of the spatial area
        strategy: "cluster" (near same sector) or "disperse" (away from others)
        
    Returns:
        Suggested (x, y) coordinates
    """
    if not all_companies:
        # If no companies, place in center
        return (size / 2, size / 2)
    
    sector_companies = [c for c in all_companies if c.sector_id == sector_id]
    
    if strategy == "cluster" and sector_companies:
        # Place near center of same-sector companies
        avg_x = sum(c.x for c in sector_companies) / len(sector_companies)
        avg_y = sum(c.y for c in sector_companies) / len(sector_companies)
        
        # Add some randomness
        noise_x = np.random.uniform(-size * 0.1, size * 0.1)
        noise_y = np.random.uniform(-size * 0.1, size * 0.1)
        
        x = np.clip(avg_x + noise_x, 0, size)
        y = np.clip(avg_y + noise_y, 0, size)
        
        return (float(x), float(y))
    
    elif strategy == "disperse":
        # Find location furthest from all companies
        best_location = None
        max_min_distance = 0.0
        
        # Try random samples
        for _ in range(100):
            test_x = np.random.uniform(0, size)
            test_y = np.random.uniform(0, size)
            
            # Find minimum distance to any existing company
            min_distance = min(
                np.sqrt((c.x - test_x)**2 + (c.y - test_y)**2)
                for c in all_companies
            )
            
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_location = (test_x, test_y)
        
        return best_location if best_location else (size / 2, size / 2)
    
    else:
        # Default: random location
        return (
            float(np.random.uniform(0, size)),
            float(np.random.uniform(0, size))
        )


def calculate_transport_cost(distance: float, base_rate: float = 0.1) -> float:
    """
    Calculate transportation cost based on distance.
    
    Args:
        distance: Distance between two locations
        base_rate: Cost per unit distance
        
    Returns:
        Total transportation cost
    """
    return distance * base_rate


def visualize_locations_ascii(all_companies: List[Company], size: float, grid_size: int = 20):
    """
    Create a simple ASCII visualization of company locations.
    
    Args:
        all_companies: List of all companies
        size: Size of the spatial area
        grid_size: Size of the ASCII grid
    """
    grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
    
    for company in all_companies:
        # Map to grid coordinates
        grid_x = int((company.x / size) * (grid_size - 1))
        grid_y = int((company.y / size) * (grid_size - 1))
        
        # Clamp to valid range
        grid_x = max(0, min(grid_size - 1, grid_x))
        grid_y = max(0, min(grid_size - 1, grid_y))
        
        # Use sector ID as marker
        marker = str(company.sector_id)
        
        # If cell already occupied, use 'X'
        if grid[grid_y][grid_x] != ' ':
            grid[grid_y][grid_x] = 'X'
        else:
            grid[grid_y][grid_x] = marker
    
    # Print the grid
    print("+" + "-" * grid_size + "+")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("+" + "-" * grid_size + "+")
    print("Legend: 0-9 = Sector ID, X = Multiple companies")


if __name__ == "__main__":
    # Example usage
    from env import IndustryEnv
    
    env = IndustryEnv(size=100.0)
    obs, _ = env.reset(seed=42, options={"initial_firms": 20})
    
    print("=== Spatial Analysis Demo ===\n")
    
    # Show ASCII visualization
    print("Company locations:")
    visualize_locations_ascii(env.companies, env.size)
    
    # Find nearest neighbors
    if env.companies:
        print(f"\nNearest neighbors to Company 0 (Sector {env.companies[0].sector_id}):")
        neighbors = find_nearest_companies(env.companies[0], env.companies, n=3)
        for i, (neighbor, dist) in enumerate(neighbors, 1):
            print(f"  {i}. Company at ({neighbor.x:.1f}, {neighbor.y:.1f}), "
                  f"Sector {neighbor.sector_id}, Distance: {dist:.2f}")
    
    # Companies in radius
    center = (50.0, 50.0)
    radius = 30.0
    nearby = find_companies_in_radius(center, env.companies, radius)
    print(f"\nCompanies within {radius} units of center ({center[0]}, {center[1]}): {len(nearby)}")
    
    # Sector clustering
    for sector_id in range(3):
        clustering = calculate_sector_clustering(env.companies, sector_id)
        if clustering != float('inf'):
            print(f"Sector {sector_id} clustering (avg distance to nearest same-sector): {clustering:.2f}")
    
    # Suggest location
    new_location = suggest_optimal_location(env.companies, sector_id=0, size=env.size, strategy="cluster")
    print(f"\nSuggested location for new Sector 0 company (cluster strategy): ({new_location[0]:.2f}, {new_location[1]:.2f})")
    
    new_location = suggest_optimal_location(env.companies, sector_id=0, size=env.size, strategy="disperse")
    print(f"Suggested location for new company (disperse strategy): ({new_location[0]:.2f}, {new_location[1]:.2f})")
