"""
Environment package for industry simulation.
Contains sector definitions, company logic, and main environment.
"""

import pickle
from pathlib import Path
from typing import Optional

from .sector import Sector, sector_relations, NUM_SECTORS, calculate_sector_tiers, SECTOR_TIERS
from .company import Company
from .env import IndustryEnv

__all__ = [
    'Sector',
    'sector_relations',
    'NUM_SECTORS',
    'calculate_sector_tiers',
    'SECTOR_TIERS',
    'Company',
    'IndustryEnv',
    'load_environment',
    'visualize_saved_environment',
]


def load_environment(filepath: str) -> IndustryEnv:
    """
    Load a saved environment from a pickle file.
    
    Args:
        filepath: Path to the saved environment file.
        
    Returns:
        IndustryEnv object with restored state.
    """
    with open(filepath, 'rb') as f:
        state = pickle.load(f)
    
    # Create a new environment with the saved config
    env_config = state.get('env_config')
    env = IndustryEnv(env_config)
    
    # Restore state
    env.companies = state['companies']
    env.num_firms = state['num_firms']
    env.current_step = state['current_step']
    env.size = state['size']
    env.max_company = state['max_company']
    env.num_sectors = state['num_sectors']
    
    return env


def visualize_saved_environment(filepath: str, output_dir: Optional[str] = None) -> None:
    """
    Load a saved environment and create interactive visualizations.
    
    Args:
        filepath: Path to the saved environment file.
        output_dir: Directory to save visualization PNGs. If None, uses './visualizations'.
    """
    import matplotlib.pyplot as plt
    from utils.visualize import create_dashboard
    
    # Load environment
    env = load_environment(filepath)
    
    # Determine output directory
    if output_dir is None:
        output_dir = 'visualizations'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create dashboard
    fig = create_dashboard(env, figsize=(16, 12))
    
    # Save as PNG
    env_filename = Path(filepath).stem
    output_path = Path(output_dir) / f"{env_filename}_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Show interactive plot with mplcursors
    plt.show()
