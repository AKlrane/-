"""
Configuration loader and manager for Industry Simulation environment.
Simplified to two main categories: Environment and Training.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union

from env.sector import NUM_SECTORS


@dataclass
class EnvironmentConfig:
    """
    Complete environment configuration.
    Contains all parameters needed to initialize and run the simulation environment.
    All values must be provided in config.json - no default values.
    """
    # Spatial and capacity parameters
    size: Optional[float] = None
    max_company: Optional[int] = None
    num_sectors: Optional[int] = None
    initial_firms: Optional[int] = None
    max_episode_steps: Optional[int] = None
    
    # Company parameters
    op_cost_rate: Optional[float] = None
    initial_capital_min: Optional[float] = None
    initial_capital_max: Optional[float] = None
    fixed_investment_amount: Optional[float] = None
    new_company_capital_min: Optional[float] = None
    new_company_capital_max: Optional[float] = None
    death_threshold: Optional[float] = None
    fixed_cost_per_step: Optional[float] = None
    
    # Supply chain parameters
    trade_volume_fraction: Optional[float] = None
    revenue_rate: Optional[float] = None
    enable_supply_chain: Optional[bool] = None
    min_distance_epsilon: Optional[float] = None
    nearest_suppliers_count: Optional[int] = None
    
    # Logistic cost parameters
    disable_logistic_costs: Optional[bool] = None
    free_delivery_distance: Optional[float] = None
    tier_logistic_cost_rate: Optional[Dict[str, float]] = None
    
    # Management cost parameters
    max_capital: Optional[float] = None
    
    # Pricing parameters - must be provided in config.json
    tier_prices: Optional[Dict[str, float]] = None
    tier_cogs: Optional[Dict[str, float]] = None
    
    # Reward parameters
    creation_reward: Optional[float] = None
    invalid_coordinate_penalty: Optional[float] = None
    invalid_tier: Optional[float] = None
    invalid_action: Optional[float] = None
    
    # Visualization parameters
    figsize_width: Optional[int] = None
    figsize_height: Optional[int] = None
    dpi: Optional[int] = None
    show_plots: Optional[bool] = None
    save_plots: Optional[bool] = None
    plot_format: Optional[str] = None
    visualize_every_n_steps: Optional[int] = None
    visualization_dir: Optional[str] = None
    
    # Product system parameters
    enable_products: Optional[bool] = None
    tier_production_ratios: Optional[Dict[str, float]] = None
    max_held_capital_rate: Optional[Dict[str, float]] = None


@dataclass
class TrainingConfig:
    """
    Complete training configuration.
    Contains all parameters needed for RL training, checkpointing, and evaluation.
    All values must be provided in config.json - no default values.
    """
    # RL framework and algorithm
    framework: Optional[str] = None
    algorithm: Optional[str] = None
    
    # Training hyperparameters
    total_timesteps: Optional[int] = None
    num_envs: Optional[int] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    gamma: Optional[float] = None
    gae_lambda: Optional[float] = None
    clip_range: Optional[float] = None
    ent_coef: Optional[float] = None
    vf_coef: Optional[float] = None
    max_grad_norm: Optional[float] = None
    n_steps: Optional[int] = None
    n_epochs: Optional[int] = None
    seed: Optional[int] = None
    verbose: Optional[int] = None
    device: Optional[str] = None
    
    # Checkpointing and logging
    log_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    save_freq: Optional[int] = None
    eval_freq: Optional[int] = None
    visualize_freq: Optional[int] = None
    n_eval_episodes: Optional[int] = None
    save_best_model: Optional[bool] = None
    
    # Advanced features
    normalize_observations: Optional[bool] = None
    normalize_rewards: Optional[bool] = None
    clip_observations: Optional[float] = None
    clip_rewards: Optional[float] = None
    frame_stack: Optional[int] = None


@dataclass
class Config:
    """Main configuration container with only two categories: Environment and Training."""
    environment: Optional[EnvironmentConfig] = None
    training: Optional[TrainingConfig] = None
    
    @classmethod
    def from_json(cls, json_path: str = "config/config.json") -> "Config":
        """
        Load configuration from JSON file.
        Supports both flat and nested (subcategorized) environment structure.
        
        Args:
            json_path: Path to config.json file (default: config/config.json)
            
        Returns:
            Config object with loaded parameters
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Config file not found: {json_path}\n"
                "All configuration must be provided in config.json - no default values."
            )
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        config = cls()
        
        # Load environment section - required
        if "environment" not in data:
            raise ValueError("'environment' section is required in config.json")
        
        env_data = data["environment"]
        
        # Check if environment has subcategories
        if any(isinstance(v, dict) for v in env_data.values()):
            # Nested structure - flatten it
            flat_env_data = {}
            for key, value in env_data.items():
                if isinstance(value, dict):
                    # Merge subcategory into flat structure
                    flat_env_data.update(value)
                else:
                    # Keep top-level values
                    flat_env_data[key] = value
            env_config = EnvironmentConfig(**flat_env_data)
        else:
            # Flat structure (backward compatible)
            env_config = EnvironmentConfig(**env_data)
        
        # Validate that all required fields are provided (not None)
        # This ensures config.json contains all necessary values
        missing_fields = []
        for field_name, field_value in env_config.__dict__.items():
            if field_value is None:
                missing_fields.append(field_name)
        
        if missing_fields:
            raise ValueError(
                f"Missing required fields in config.json environment section: {', '.join(missing_fields)}\n"
                "All configuration values must be provided in config.json - no default values."
            )
        
        config.environment = env_config
        
        # Load training section - required
        if "training" not in data:
            raise ValueError("'training' section is required in config.json")
        
        train_config = TrainingConfig(**data["training"])
        
        # Validate that all required fields are provided (not None)
        missing_fields = []
        for field_name, field_value in train_config.__dict__.items():
            if field_value is None:
                missing_fields.append(field_name)
        
        if missing_fields:
            raise ValueError(
                f"Missing required fields in config.json training section: {', '.join(missing_fields)}\n"
                "All configuration values must be provided in config.json - no default values."
            )
        
        config.training = train_config
        
        return config
    
    def to_json(self, json_path: str = "config_output.json"):
        """
        Save configuration to JSON file.
        
        Args:
            json_path: Path to output JSON file
        """
        data = {
            "environment": self.environment.__dict__,
            "training": self.training.__dict__,
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Configuration saved to: {json_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment.__dict__,
            "training": self.training.__dict__,
        }
    
    def print_summary(self):
        """Print a summary of the configuration."""
        print("=" * 70)
        print("CONFIGURATION SUMMARY")
        print("=" * 70)
        
        print("\n🌍 Environment:")
        print(f"   Size: {self.environment.size}")
        print(f"   Max companies: {self.environment.max_company}")
        print(f"   Sectors: {self.environment.num_sectors}")
        print(f"   Initial firms: {self.environment.initial_firms}")
        print(f"   Max episode steps: {self.environment.max_episode_steps}")
        
        print("\n🏢 Company Parameters:")
        print(f"   Operating cost rate: {self.environment.op_cost_rate}")
        print(f"   Initial capital range: ${self.environment.initial_capital_min:,.0f} - ${self.environment.initial_capital_max:,.0f}")
        print(f"   Death threshold: ${self.environment.death_threshold:,.0f}")
        
        print("\n🚚 Supply Chain:")
        print(f"   Enabled: {self.environment.enable_supply_chain}")
        print(f"   Trade volume fraction: {self.environment.trade_volume_fraction}")
        print(f"   Revenue rate: {self.environment.revenue_rate}")
        print(f"   Logistic costs disabled: {self.environment.disable_logistic_costs}")
        
        print("\n🎁 Rewards:")
        print(f"   Creation reward: {self.environment.creation_reward}")
        print(f"   Invalid coordinate penalty: {self.environment.invalid_coordinate_penalty}")
        print(f"   Invalid tier penalty: {self.environment.invalid_tier}")
        print(f"   Invalid action penalty: {self.environment.invalid_action}")
        
        print("\n🤖 Training:")
        print(f"   Framework: {self.training.framework}")
        print(f"   Algorithm: {self.training.algorithm}")
        print(f"   Total timesteps: {self.training.total_timesteps:,}")
        print(f"   Learning rate: {self.training.learning_rate}")
        print(f"   Number of envs: {self.training.num_envs}")
        print(f"   Gamma: {self.training.gamma}")
        print(f"   Batch size: {self.training.batch_size}")
        
        print("\n� Checkpointing:")
        print(f"   Log directory: {self.training.log_dir}")
        print(f"   Checkpoint directory: {self.training.checkpoint_dir}")
        print(f"   Save frequency: every {self.training.save_freq:,} steps")
        print(f"   Eval frequency: every {self.training.eval_freq:,} steps")
        
        
        print("=" * 70)


def load_config(config_path: str = "config.json") -> Config:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to config.json file (default: config.json)
                    If relative path, looks in config/ directory
        
    Returns:
        Config object
    """
    # If path is relative and doesn't exist, try looking in config directory
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        config_dir = os.path.dirname(__file__)
        config_path = os.path.join(config_dir, config_path)
    
    return Config.from_json(config_path)


def create_default_config(output_path: str = "config_default.json"):
    """
    Create a default configuration file.
    
    Args:
        output_path: Where to save the default config
                    If relative path, saves to config/ directory
    """
    # If path is relative, save to config directory
    if not os.path.isabs(output_path):
        config_dir = os.path.dirname(__file__)
        output_path = os.path.join(config_dir, output_path)
    
    config = Config()
    config.to_json(output_path)
    print(f"✅ Default configuration created: {output_path}")


if __name__ == "__main__":
    # Demo: Load and print configuration
    print("Loading configuration from config/config.json...")
    config = load_config()  # Uses default path: config/config.json
    config.print_summary()
    
    # Demo: Create environment with config
    print("\n" + "=" * 70)
    print("EXAMPLE: Creating environment with config")
    print("=" * 70)
    
    print(f"""
from config import load_config
from env import IndustryEnv

# Load config (defaults to config/config.json)
config = load_config()

# Create environment with config object
env = IndustryEnv(config.environment)

# Reset with initial firms from config
obs, info = env.reset(options={{"initial_firms": config.environment.initial_firms}})

print(f"Environment created with {{obs['num_firms']}} initial firms")
""")
    
    # Demo: Save modified config
    print("\n" + "=" * 70)
    print("EXAMPLE: Modifying and saving config")
    print("=" * 70)
    
    # Modify some parameters
    config.training.total_timesteps = 500000
    config.environment.disable_logistic_costs = True
    
    # Save modified config (will be saved to config/ directory)
    config_dir = os.path.dirname(__file__)
    modified_path = os.path.join(config_dir, "config_modified.json")
    config.to_json(modified_path)
    
    print(f"\nModified configuration saved to {modified_path}")
    print(f"   Total timesteps: {config.training.total_timesteps:,}")
    print(f"   Logistic costs disabled: {config.environment.disable_logistic_costs}")
