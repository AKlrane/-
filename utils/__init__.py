"""
Utility functions for industry simulation.
Contains visualization and spatial analysis utilities.
"""

from .visualize import (
    plot_companies,
    plot_sector_distribution,
    plot_capital_distribution,
    plot_sector_clusters,
    plot_network_connections,
    create_dashboard,
)

from .spatial_utils import (
    find_nearest_companies,
    find_companies_in_radius,
    calculate_sector_clustering,
    suggest_optimal_location,
    visualize_locations_ascii,
)

__all__ = [
    # Visualization
    'plot_companies',
    'plot_sector_distribution',
    'plot_capital_distribution',
    'plot_sector_clusters',
    'plot_network_connections',
    'create_dashboard',
    # Spatial utilities
    'find_nearest_companies',
    'find_companies_in_radius',
    'calculate_sector_clustering',
    'suggest_optimal_location',
    'visualize_locations_ascii',
]
