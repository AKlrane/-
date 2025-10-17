"""
Configuration loader and manager for Industry Simulation environment.
Simplified to two main categories: Environment and Training.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from env.sector import NUM_SECTORS


@dataclass
class EnvironmentConfig:
    """
    Complete environment configuration.
    Contains all parameters needed to initialize and run the simulation environment.
    """
    # Spatial and capacity parameters
    size: float = 100.0
    max_company: int = 1000
    num_sectors: int = NUM_SECTORS
    initial_firms: int = 10
    max_episode_steps: int = 1000
    
    # Action parameters
    max_actions_per_step: int = 10
    
    # Company parameters
    op_cost_rate: float = 0.05
    initial_capital_min: float = 10000.0
    initial_capital_max: float = 100000.0
    investment_min: float = 0.0
    investment_max: float = 1000000.0
    new_company_capital_min: float = 1000.0
    new_company_capital_max: float = 1000000.0
    death_threshold: float = 0.0
    fixed_cost_per_step: float = -5.0
    
    # Supply chain parameters
    trade_volume_fraction: float = 0.01
    revenue_rate: float = 1.0
    enable_supply_chain: bool = True
    min_distance_epsilon: float = 0.1
    nearest_suppliers_count: int = 5
    
    # Logistic cost parameters
    logistic_cost_rate: float = 1.0
    
    # Management cost parameters
    max_capital: float = 100000000.0  # 1 billion - threshold for logarithmic cost
    
    # Pricing parameters
    tier_prices: dict = field(default_factory=lambda: {
        "Raw": 10.0,
        "Parts": 50.0,
        "Electronics": 120.0,
        "Battery/Motor": 350.0,
        "OEM": 4500.0,
        "Service": 5500.0,
        "Other": 10.5
    })
    tier_cogs: dict = field(default_factory=lambda: {
        "Raw": 9.0,
        "Other": 10.0
    })
    
    # Reward parameters
    investment_multiplier: float = 0.01
    creation_reward: float = 50.0
    invalid_action_penalty: float = -20.0
    invalid_firm_penalty: float = -10.0
    profit_multiplier: float = 0.001
    
    # Visualization parameters
    figsize_width: int = 18
    figsize_height: int = 12
    dpi: int = 600
    show_plots: bool = False
    save_plots: bool = True
    plot_format: str = "png"
    visualize_every_n_steps: int = 0  # 0 = disable periodic visualization, >0 = visualize every N steps
    visualization_dir: str = "visualizations"  # Directory to save visualizations
    
    # Ablation switches
    disable_logistic_costs: bool = False
    disable_supply_chain: bool = False
    fixed_locations: bool = False
    allow_negative_capital: bool = False
    
    # Product system parameters
    enable_products: bool = True
    tier_production_ratios: dict = field(default_factory=lambda: {
        "Raw": 0.5,
        "Parts": 0.3,
        "Electronics": 0.3,
        "Battery/Motor": 0.3,
        "OEM": 0.2,
        "Service": 0.1,
        "Other": 0.1
    })
    max_held_capital_rate: dict = field(default_factory=lambda: {
        "Raw": 0.3,
        "Parts": 0.4,
        "Electronics": 0.4,
        "Battery/Motor": 0.4,
        "OEM": 0.3,
        "Service": 0.2,
        "Other": 0.3
    })


@dataclass
class TrainingConfig:
    """
    Complete training configuration.
    Contains all parameters needed for RL training, checkpointing, and evaluation.
    """
    # RL framework and algorithm
    framework: str = "sb3"
    algorithm: str = "ppo"
    
    # Training hyperparameters
    total_timesteps: int = 100000
    num_envs: int = 4
    learning_rate: float = 0.0003
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_steps: int = 2048
    n_epochs: int = 10
    seed: int = 42
    verbose: int = 1
    device: str = "auto"  # "auto", "cuda", "cpu", "cuda:0", "cuda:1", etc.
    
    # Checkpointing and logging
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    save_freq: int = 10000
    eval_freq: int = 5000
    visualize_freq: int = 10000
    n_eval_episodes: int = 10
    save_best_model: bool = True
    
    # Advanced features
    normalize_observations: bool = False
    normalize_rewards: bool = False
    clip_observations: float = 10.0
    clip_rewards: float = 10.0
    frame_stack: int = 1


@dataclass
class Config:
    """Main configuration container with only two categories: Environment and Training."""
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
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
            print(f"⚠️  Config file not found: {json_path}")
            print("   Using default configuration")
            return cls()
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        config = cls()
        
        # Load environment section
        if "environment" in data:
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
                config.environment = EnvironmentConfig(**flat_env_data)
            else:
                # Flat structure (backward compatible)
                config.environment = EnvironmentConfig(**env_data)
        
        # Load training section
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        
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
        print(f"   Logistic cost rate: {self.environment.logistic_cost_rate}")
        
        print("\n🎁 Rewards:")
        print(f"   Creation reward: {self.environment.creation_reward}")
        print(f"   Invalid action penalty: {self.environment.invalid_action_penalty}")
        print(f"   Profit multiplier: {self.environment.profit_multiplier}")
        
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
        
        print("\n🔬 Ablation Studies:")
        if self.environment.disable_logistic_costs:
            print("   ⚠️  Logistic costs DISABLED")
        if self.environment.disable_supply_chain:
            print("   ⚠️  Supply chain DISABLED")
        if not any([self.environment.disable_logistic_costs, self.environment.disable_supply_chain]):
            print("   All features enabled")
        
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
    config.environment.logistic_cost_rate = 200.0
    
    # Save modified config (will be saved to config/ directory)
    config_dir = os.path.dirname(__file__)
    modified_path = os.path.join(config_dir, "config_modified.json")
    config.to_json(modified_path)
    
    print(f"\nModified configuration saved to {modified_path}")
    print(f"   Total timesteps: {config.training.total_timesteps:,}")
    print(f"   Logistic cost rate: {config.environment.logistic_cost_rate}")
