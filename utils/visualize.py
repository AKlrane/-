"""
Visualization utilities for the Industry Simulation Environment.
Uses matplotlib to create interactive visualizations of company locations, sectors, and metrics.
"""

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Optional, Tuple
import mplcursors
from env import Company, sector_relations, IndustryEnv


# Color scheme for automotive supply chain sectors (7 sectors total)
# Based on 5-tier automotive supply chain model
SECTOR_COLORS = [
    '#8B4513',  # 0: Raw Materials - Brown (extraction/processing)
    '#FF6B6B',  # 1: Parts - Red (component manufacturing)
    '#45B7D1',  # 2: Electronics - Blue (tech components)
    '#FFEAA7',  # 3: Battery/Motor - Yellow (power systems)
    '#96CEB4',  # 4: OEM - Green (vehicle assembly)
    '#A29BFE',  # 5: Service - Purple (retail/maintenance)
    '#DFE6E9',  # 6: Other - Light Gray (miscellaneous)
]


def plot_companies(
    companies: List[Company],
    size: float = 100.0,
    title: str = "Company Location Distribution",
    show_labels: bool = False,
    figsize: Tuple[int, int] = (12, 10)
) -> Optional[Figure]:
    """
    Create a scatter plot of company locations colored by sector.
    
    Args:
        companies: List of Company objects to visualize
        size: Size of the spatial area (for axis limits)
        title: Plot title
        show_labels: Whether to show company ID labels on hover
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    if not companies:
        print("No companies to visualize!")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    x_coords = [c.x for c in companies]
    y_coords = [c.y for c in companies]
    colors = [SECTOR_COLORS[c.sector_id % len(SECTOR_COLORS)] for c in companies]
    sizes = [np.sqrt(c.capital) / 10 for c in companies]  # Size proportional to sqrt of capital
    
    # Create scatter plot
    scatter = ax.scatter(
        x_coords, 
        y_coords, 
        c=colors, 
        s=sizes, 
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Set axis properties (map centered at origin)
    ax.set_xlim(-size/2, size/2)
    ax.set_ylim(-size/2, size/2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Create legend for sectors
    legend_elements = []
    sector_ids = sorted(set(c.sector_id for c in companies))
    for sector_id in sector_ids:
        if sector_id < len(sector_relations):
            sector_name = sector_relations[sector_id].name
            color = SECTOR_COLORS[sector_id % len(SECTOR_COLORS)]
            legend_elements.append(
                mpatches.Patch(color=color, label=f'Sector {sector_id}: {sector_name}')
            )
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    # Add interactive hover functionality
    if show_labels:
        cursor = mplcursors.cursor(scatter, hover=True)
        
        @cursor.connect("add")
        def on_add(sel):
            idx = sel.index
            company = companies[idx]
            sector_name = sector_relations[company.sector_id].name if company.sector_id < len(sector_relations) else "Unknown"
            text = (
                f"Company {idx}\n"
                f"Sector: {sector_name}\n"
                f"Location: ({company.x:.1f}, {company.y:.1f})\n"
                f"Capital: ${company.capital:,.0f}\n"
                f"Revenue: ${company.revenue:,.0f}"
            )
            sel.annotation.set(text=text)
            sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)
    
    plt.tight_layout()
    return fig


def plot_sector_distribution(
    companies: List[Company],
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[Figure]:
    """
    Create a bar chart showing the distribution of companies across sectors.
    
    Args:
        companies: List of Company objects
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    if not companies:
        print("No companies to visualize!")
        return None
    
    # Count companies per sector
    sector_counts = {}
    for company in companies:
        sector_counts[company.sector_id] = sector_counts.get(company.sector_id, 0) + 1
    
    # Sort by sector ID
    sorted_sectors = sorted(sector_counts.items())
    sector_ids = [s[0] for s in sorted_sectors]
    counts = [s[1] for s in sorted_sectors]
    
    # Get sector names
    sector_names = []
    for sid in sector_ids:
        if sid < len(sector_relations):
            sector_names.append(f"{sector_relations[sid].name}")
        else:
            sector_names.append(f"Sector {sid}")
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=figsize)
    colors = [SECTOR_COLORS[sid % len(SECTOR_COLORS)] for sid in sector_ids]
    bars = ax.bar(sector_names, counts, color=colors, alpha=0.7, edgecolor='black')
    
    # Customize plot
    ax.set_xlabel('Sector', fontsize=12)
    ax.set_ylabel('Number of Companies', fontsize=12)
    ax.set_title('Company Distribution by Sector', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    return fig


def plot_capital_distribution(
    companies: List[Company],
    size: float = 100.0,
    grid_size: int = 10,
    figsize: Tuple[int, int] = (10, 8)
) -> Optional[Figure]:
    """
    Create a heatmap showing total capital distribution across the spatial area.
    
    Args:
        companies: List of Company objects
        size: Size of the spatial area
        grid_size: Number of grid cells per dimension
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    if not companies:
        print("No companies to visualize!")
        return None
    
    # Create grid for capital accumulation
    capital_grid = np.zeros((grid_size, grid_size))
    
    for company in companies:
        # Map to grid coordinates (map centered at origin)
        # Transform from [-size/2, size/2] to [0, grid_size]
        grid_x = int(((company.x + size/2) / size) * grid_size)
        grid_y = int(((company.y + size/2) / size) * grid_size)
        
        # Clamp to valid range
        grid_x = max(0, min(grid_size - 1, grid_x))
        grid_y = max(0, min(grid_size - 1, grid_y))
        
        capital_grid[grid_y, grid_x] += company.capital
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        capital_grid,
        cmap='YlOrRd',
        interpolation='bilinear',
        origin='lower',
        extent=(-size/2, size/2, -size/2, size/2)  # Map centered at origin
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Total Capital ($)', rotation=270, labelpad=20, fontsize=12)
    
    # Overlay company positions
    x_coords = [c.x for c in companies]
    y_coords = [c.y for c in companies]
    ax.scatter(x_coords, y_coords, c='blue', s=20, alpha=0.5, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title('Capital Distribution Heatmap', fontsize=14, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig


def plot_sector_clusters(
    companies: List[Company],
    size: float = 100.0,
    figsize: Tuple[int, int] = (12, 10)
) -> Optional[Figure]:
    """
    Create subplots showing spatial distribution for each sector.
    
    Args:
        companies: List of Company objects
        size: Size of the spatial area
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    if not companies:
        print("No companies to visualize!")
        return None
    
    # Get unique sectors
    sector_ids = sorted(set(c.sector_id for c in companies))
    n_sectors = len(sector_ids)
    
    # Calculate grid layout
    n_cols = 3
    n_rows = (n_sectors + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_sectors > 1 else [axes]
    
    for idx, sector_id in enumerate(sector_ids):
        ax = axes[idx]
        
        # Get companies in this sector
        sector_companies = [c for c in companies if c.sector_id == sector_id]
        x_coords = [c.x for c in sector_companies]
        y_coords = [c.y for c in sector_companies]
        sizes = [np.sqrt(c.capital) / 10 for c in sector_companies]
        
        # Plot
        color = SECTOR_COLORS[sector_id % len(SECTOR_COLORS)]
        ax.scatter(x_coords, y_coords, c=color, s=sizes, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Customize subplot (map centered at origin)
        ax.set_xlim(-size/2, size/2)
        ax.set_ylim(-size/2, size/2)
        ax.set_aspect('equal', adjustable='box')
        sector_name = sector_relations[sector_id].name if sector_id < len(sector_relations) else f"Sector {sector_id}"
        ax.set_title(f'{sector_name} (n={len(sector_companies)})', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Hide unused subplots
    for idx in range(n_sectors, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('Sector Clustering Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_network_connections(
    companies: List[Company],
    size: float = 100.0,
    max_distance: float = 30.0,
    figsize: Tuple[int, int] = (12, 10)
) -> Optional[Figure]:
    """
    Visualize potential connections between nearby companies.
    
    Args:
        companies: List of Company objects
        size: Size of the spatial area
        max_distance: Maximum distance for drawing connections
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    if not companies:
        print("No companies to visualize!")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw connections first (so they appear behind points)
    for i, company1 in enumerate(companies):
        for j, company2 in enumerate(companies):
            if i < j:  # Avoid duplicate connections
                distance = company1.distance_to(company2)
                if distance <= max_distance:
                    # Draw line with alpha based on distance
                    alpha = 1.0 - (distance / max_distance)
                    ax.plot(
                        [company1.x, company2.x],
                        [company1.y, company2.y],
                        'gray',
                        alpha=alpha * 0.3,
                        linewidth=0.5
                    )
    
    # Draw companies on top
    x_coords = [c.x for c in companies]
    y_coords = [c.y for c in companies]
    colors = [SECTOR_COLORS[c.sector_id % len(SECTOR_COLORS)] for c in companies]
    sizes = [np.sqrt(c.capital) / 10 for c in companies]
    
    ax.scatter(
        x_coords,
        y_coords,
        c=colors,
        s=sizes,
        alpha=0.7,
        edgecolors='black',
        linewidth=1
    )
    
    # Customize plot (map centered at origin)
    ax.set_xlim(-size/2, size/2)
    ax.set_ylim(-size/2, size/2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(f'Company Network (connections within {max_distance} units)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


def create_dashboard(
    env: IndustryEnv,
    figsize: Tuple[int, int] = (18, 12)
) -> Optional[Figure]:
    """
    Create a comprehensive dashboard with multiple visualizations.
    
    Args:
        env: IndustryEnv object
        figsize: Figure size (width, height)
        
    Returns:
        matplotlib Figure object
    """
    if not env.companies:
        print("No companies to visualize!")
        return None
    
    fig = Figure(figsize=figsize)
    
    # Create grid layout with better spacing
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25, 
                         left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    # 1. Main location plot (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    x_coords = [c.x for c in env.companies]
    y_coords = [c.y for c in env.companies]
    colors = [SECTOR_COLORS[c.sector_id % len(SECTOR_COLORS)] for c in env.companies]
    sizes = [np.sqrt(c.capital) / 10 for c in env.companies]
    
    ax1.scatter(x_coords, y_coords, c=colors, s=sizes, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax1.set_xlim(-env.size/2, env.size/2)
    ax1.set_ylim(-env.size/2, env.size/2)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title('Company Locations', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 2. Sector distribution (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    sector_counts = {}
    for company in env.companies:
        sector_counts[company.sector_id] = sector_counts.get(company.sector_id, 0) + 1
    
    sorted_sectors = sorted(sector_counts.items())
    sector_ids = [s[0] for s in sorted_sectors]
    counts = [s[1] for s in sorted_sectors]
    sector_names = [sector_relations[sid].name if sid < len(sector_relations) else f"S{sid}" for sid in sector_ids]
    colors_bar = [SECTOR_COLORS[sid % len(SECTOR_COLORS)] for sid in sector_ids]
    
    ax2.barh(sector_names, counts, color=colors_bar, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Number of Companies')
    ax2.set_title('Sector Distribution', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # 3. Capital by sector (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    sector_capital = {}
    for company in env.companies:
        sector_capital[company.sector_id] = sector_capital.get(company.sector_id, 0) + company.capital
    
    sorted_capital = sorted(sector_capital.items())
    sector_ids_cap = [s[0] for s in sorted_capital]
    capitals = [s[1] for s in sorted_capital]
    sector_names_cap = [sector_relations[sid].name if sid < len(sector_relations) else f"S{sid}" for sid in sector_ids_cap]
    colors_cap = [SECTOR_COLORS[sid % len(SECTOR_COLORS)] for sid in sector_ids_cap]
    
    ax3.bar(sector_names_cap, capitals, color=colors_cap, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Total Capital ($)')
    ax3.set_title('Capital by Sector', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Statistics (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Calculate statistics
    total_capital = sum(c.capital for c in env.companies)
    total_revenue = sum(c.revenue for c in env.companies)
    avg_capital = total_capital / len(env.companies)
    avg_revenue = total_revenue / len(env.companies)
    max_capital = max(c.capital for c in env.companies)
    min_capital = min(c.capital for c in env.companies)
    
    stats_text = f"""
    ENVIRONMENT STATISTICS
    {'='*45}
    
    Total Companies:      {len(env.companies)}
    Current Step:         {env.current_step}
    Sectors Active:       {len(sector_counts)}
    
    {'='*45}
    CAPITAL METRICS
    {'='*45}
    
    Total Capital:        ${total_capital:,.0f}
    Average Capital:      ${avg_capital:,.0f}
    Max Capital:          ${max_capital:,.0f}
    Min Capital:          ${min_capital:,.0f}
    
    {'='*45}
    REVENUE METRICS
    {'='*45}
    
    Total Revenue:        ${total_revenue:,.0f}
    Average Revenue:      ${avg_revenue:,.0f}
    
    {'='*45}
    """
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4, pad=1.0))
    
    # Add title with better positioning
    fig.suptitle('Industry Simulation Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    return fig


if __name__ == "__main__":
    # Example usage
    print("=== Visualization Demo ===\n")
    
    # Create environment with some companies
    env = IndustryEnv()
    obs, _ = env.reset(seed=42, options={"initial_firms": 30})
    
    print(f"Created environment with {len(env.companies)} companies\n")
    
    # 1. Basic location plot
    print("1. Creating company location plot...")
    fig1 = plot_companies(env.companies, size=env.size, show_labels=True, 
                          title="Company Locations by Sector")
    
    # 2. Sector distribution
    print("2. Creating sector distribution plot...")
    fig2 = plot_sector_distribution(env.companies)
    
    # 3. Capital heatmap
    print("3. Creating capital distribution heatmap...")
    fig3 = plot_capital_distribution(env.companies, size=env.size)
    
    # 4. Sector clusters
    print("4. Creating sector clustering analysis...")
    fig4 = plot_sector_clusters(env.companies, size=env.size)
    
    # 5. Network connections
    print("5. Creating network connection plot...")
    fig5 = plot_network_connections(env.companies, size=env.size, max_distance=25.0)
    
    # 6. Dashboard
    print("6. Creating comprehensive dashboard...")
    fig6 = create_dashboard(env)
    
    print("\nAll visualizations created!")
    print("Close the plot windows to exit.")
    
    plt.show()
