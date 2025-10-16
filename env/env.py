import gymnasium as gym
import numpy as np
import pickle
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

from .sector import sector_relations, NUM_SECTORS, SECTOR_TIERS
from .company import Company


class IndustryEnv(gym.Env):
    """
    Gymnasium environment for industry simulation with multiple companies and sectors.

    Actions:
        - op=0: Invest in existing company (increases capital)
        - op=1: Create new company in a specific sector

    Observations:
        - Number of active firms
        - Company details (capital, sector, revenue)
    """

    def __init__(self, env_config=None):
        """
        Initialize the Industry Simulation environment.

        Args:
            env_config: EnvironmentConfig object from config.py, or None to use defaults
        """
        super().__init__()

        # If config object provided, use it; otherwise use default values
        if env_config is not None:
            # Spatial and capacity
            self.size = env_config.size
            self.max_company = env_config.max_company
            self.num_sectors = env_config.num_sectors
            self.max_episode_steps = env_config.max_episode_steps

            # Action parameters
            self.max_actions_per_step = env_config.max_actions_per_step

            # Company parameters
            self.op_cost_rate = env_config.op_cost_rate
            self.initial_capital_min = env_config.initial_capital_min
            self.initial_capital_max = env_config.initial_capital_max
            self.investment_min = env_config.investment_min
            self.investment_max = env_config.investment_max
            self.new_company_capital_min = env_config.new_company_capital_min
            self.new_company_capital_max = env_config.new_company_capital_max
            self.death_threshold = env_config.death_threshold

            # Supply chain
            self.enable_supply_chain = env_config.enable_supply_chain
            self.trade_volume_fraction = env_config.trade_volume_fraction
            self.revenue_rate = env_config.revenue_rate
            self.min_distance_epsilon = env_config.min_distance_epsilon

            # Logistics
            self.logistic_cost_rate = env_config.logistic_cost_rate
            self.disable_logistic_costs = env_config.disable_logistic_costs

            # Rewards
            self.investment_multiplier = env_config.investment_multiplier
            self.creation_reward = env_config.creation_reward
            self.invalid_action_penalty = env_config.invalid_action_penalty
            self.invalid_firm_penalty = env_config.invalid_firm_penalty
            self.profit_multiplier = env_config.profit_multiplier

            # Ablation
            self.disable_supply_chain = env_config.disable_supply_chain
            self.fixed_locations = env_config.fixed_locations
            self.allow_negative_capital = env_config.allow_negative_capital

            # Product system parameters
            self.enable_products = getattr(env_config, "enable_products", True)
            self.production_capacity_ratio = getattr(
                env_config, "production_capacity_ratio", 0.1
            )
            self.purchase_budget_ratio = getattr(
                env_config, "purchase_budget_ratio", 0.2
            )
            
            # Visualization parameters
            self.visualize_every_n_steps = getattr(env_config, "visualize_every_n_steps", 0)
            self.visualization_dir = getattr(env_config, "visualization_dir", "visualizations")
            self.figsize_width = getattr(env_config, "figsize_width", 12)
            self.figsize_height = getattr(env_config, "figsize_height", 8)
            self.dpi = getattr(env_config, "dpi", 150)
            self.save_plots = getattr(env_config, "save_plots", True)
            self.show_plots = getattr(env_config, "show_plots", False)
            self.plot_format = getattr(env_config, "plot_format", "png")
            
            # Store config for saving
            self.env_config = env_config
        else:
            # Default values for backward compatibility
            self.size = 100.0
            self.max_company = 1000
            self.num_sectors = NUM_SECTORS
            self.max_episode_steps = 1000
            self.max_actions_per_step = 10
            self.op_cost_rate = 0.05
            self.initial_capital_min = 10000.0
            self.initial_capital_max = 100000.0
            self.investment_min = 0.0
            self.investment_max = 1000000.0
            self.new_company_capital_min = 1000.0
            self.new_company_capital_max = 1000000.0
            self.death_threshold = 0.0
            self.enable_supply_chain = True
            self.trade_volume_fraction = 0.01
            self.revenue_rate = 1.0
            self.min_distance_epsilon = 0.1
            self.logistic_cost_rate = 1.0
            self.disable_logistic_costs = False
            self.investment_multiplier = 0.01
            self.creation_reward = 50.0
            self.invalid_action_penalty = -20.0
            self.invalid_firm_penalty = -10.0
            self.profit_multiplier = 0.001
            self.disable_supply_chain = False
            self.fixed_locations = False
            self.allow_negative_capital = False

            # Product system defaults
            self.enable_products = True
            self.production_capacity_ratio = 0.1
            self.purchase_budget_ratio = 0.2
            
            # Visualization defaults
            self.visualize_every_n_steps = 0
            self.visualization_dir = "visualizations"
            self.figsize_width = 18
            self.figsize_height = 12
            self.dpi = 600
            self.save_plots = True
            self.show_plots = False
            self.plot_format = "png"
            self.env_config = None

        # Create visualization directory if needed
        if self.visualize_every_n_steps > 0 and self.save_plots:
            Path(self.visualization_dir).mkdir(parents=True, exist_ok=True)

        # Track companies
        self.companies: list[Company] = []
        self.num_firms = 0

        # Observation space: simplified to track firm count and aggregate metrics
        self.observation_space = gym.spaces.Dict(
            {
                "num_firms": gym.spaces.Discrete(self.max_company + 1),
                "total_capital": gym.spaces.Box(
                    low=0.0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "sector_counts": gym.spaces.Box(
                    low=0,
                    high=self.max_company,
                    shape=(self.num_sectors,),
                    dtype=np.int32,
                ),
                "avg_revenue": gym.spaces.Box(
                    low=0.0, high=np.inf, shape=(1,), dtype=np.float32
                ),
            }
        )

        # Simplified continuous action space for single action per step
        # Box action space with 4 continuous components:
        #   [0]: action_type (0.0-1.0, <0.5=invest, >=0.5=create)
        #   [1]: amount (continuous value in normalized range 0-1)
        #   [2]: x coordinate (continuous value in range -20 to 20 km)
        #   [3]: y coordinate (continuous value in range -20 to 20 km)
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, -20.0, -20.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 20.0, 20.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Map range: ±20 km
        self.map_min = -20.0
        self.map_max = 20.0

        self.current_step = 0

    def _build_supply_chain_network(self):
        """
        Build supply chain relationships between companies based on their tiers.
        Lower tier companies supply to higher tier companies.
        """
        # Clear existing relationships
        for company in self.companies:
            company.suppliers = []
            company.customers = []

        # Build relationships based on tiers
        for company in self.companies:
            # Find suppliers (companies in lower tiers)
            for other in self.companies:
                if other == company:
                    continue

                # If other company is in a lower tier, it can be a supplier
                if other.tier < company.tier:
                    company.suppliers.append(other)
                    other.customers.append(company)

    def _simulate_supply_chain(self):
        """
        Simulate product-based supply chain interactions.

        Process for each step:
        1. Tier 0 (most upstream) companies produce products
        2. For tiers 1 to max:
           - Companies calculate their needs based on customer orders
           - Companies purchase from suppliers (limited by budget)
           - Companies fulfill customer orders from inventory

        Respects disable_supply_chain and enable_products flags.
        """
        # Skip if supply chain or products are disabled
        if (
            self.disable_supply_chain
            or not self.enable_supply_chain
            or not self.enable_products
        ):
            # Fall back to old generic supply chain simulation
            self._simulate_generic_supply_chain()
            return

        # Reset step counters for all companies
        for company in self.companies:
            company.reset_step_counters()

        # Step 1: Tier 0 companies produce products
        tier_0_companies = [c for c in self.companies if c.tier == 0]
        for company in tier_0_companies:
            company.produce_products()

        # Step 2: Process each tier from 1 to max, purchasing and selling
        max_tier = max(c.tier for c in self.companies) if self.companies else 0

        for current_tier in range(1, max_tier + 1):
            tier_companies = [c for c in self.companies if c.tier == current_tier]

            for company in tier_companies:
                # Calculate total orders from downstream customers
                # Each customer wants to order based on their purchase budget
                total_customer_orders = 0.0
                for customer in company.customers:
                    order_amount = (
                        customer.get_max_purchase_budget() / len(customer.suppliers)
                        if customer.suppliers
                        else 0.0
                    )
                    total_customer_orders += order_amount

                # Purchase from upstream suppliers to fulfill orders + maintain inventory
                company.purchase_from_suppliers(total_customer_orders, self.disable_logistic_costs)

    def _simulate_generic_supply_chain(self):
        """
        Original generic supply chain simulation (fallback when products disabled).
        Companies in supplier sectors trade with companies in consumer sectors.
        Logistic costs are applied based on distance (inverse square law).
        """
        # For each company, find potential trading partners based on sector relationships
        for company in self.companies:
            sector = sector_relations[company.sector_id]

            # Find companies in consumer sectors that this company can supply to
            potential_customers = []
            for other in self.companies:
                if company == other:
                    continue
                other_sector = sector_relations[other.sector_id]
                # Check if this company's sector is in the other's suppliers list
                # or if this sector supplies "All Sectors"
                if (
                    sector.name in other_sector.suppliers
                    or "All Sectors" in sector.consumers
                ):
                    potential_customers.append(other)

            # If there are customers, simulate trade
            if potential_customers:
                # Use configured trade volume fraction
                trade_volume_per_customer = (
                    company.capital
                    * self.trade_volume_fraction
                    / len(potential_customers)
                )

                for customer in potential_customers:
                    # Calculate logistic cost based on distance (inverse square law)
                    # Only apply if logistic costs are not disabled
                    if not self.disable_logistic_costs:
                        logistic_cost = company.calculate_logistic_cost_to(
                            customer, trade_volume_per_customer
                        )
                        # Apply logistic cost to the supplier (company bears transportation cost)
                        company.add_logistic_cost(logistic_cost)

                    # Add revenue to supplier
                    company.add_revenue(trade_volume_per_customer)

                    # Customer receives goods (simplified - no cost to customer in this model)
                    # In a more complex model, customer might also bear some logistics costs

    def _get_observation(self) -> Dict[str, int | np.ndarray]:
        """Generate current observation from environment state."""
        if self.num_firms == 0:
            return {
                "num_firms": 0,
                "total_capital": np.array([0.0], dtype=np.float32),
                "sector_counts": np.zeros(self.num_sectors, dtype=np.int32),
                "avg_revenue": np.array([0.0], dtype=np.float32),
            }

        total_capital = sum(c.capital for c in self.companies)
        total_revenue = sum(c.revenue for c in self.companies)
        avg_revenue = total_revenue / self.num_firms if self.num_firms > 0 else 0.0

        sector_counts = np.zeros(self.num_sectors, dtype=np.int32)
        for company in self.companies:
            sector_counts[company.sector_id] += 1

        return {
            "num_firms": self.num_firms,
            "total_capital": np.array([total_capital], dtype=np.float32),
            "sector_counts": sector_counts,
            "avg_revenue": np.array([avg_revenue], dtype=np.float32),
        }

    def _calculate_reward(self, info: Dict[str, Any]) -> float:
        """
        Calculate the total reward for the current step.

        Args:
            info: Information dict containing:
                - action_result: Result of the single action taken
                - action_type: Type of action (0=invest, 1=create)
                - num_valid_actions: 1 if successful, 0 if failed
                - num_invalid_actions: 0 if successful, 1 if failed
                - investment_amount: Amount invested (if applicable)
                - total_profit: Total profit from all companies
                - total_logistic_cost: Total logistic costs
                - num_firms: Current number of firms
                - num_deaths: Number of companies that died

        Returns:
            Total reward combining action rewards and profit-based reward
        """
        reward = 0.0

        # Calculate action-specific reward
        action_result = info.get("action_result", "unknown")

        if action_result == "valid_invest":
            # Reward for successful investment
            investment_amount = info.get("investment_amount", 0.0)
            reward += investment_amount * self.investment_multiplier

        elif action_result == "valid_create":
            # Reward for successful company creation
            reward += self.creation_reward

        elif action_result in ["invalid_firm", "invalid_no_firms"]:
            # Penalty for invalid firm selection
            reward += self.invalid_firm_penalty

        elif action_result in [
            "invalid_amount",
            "invalid_location",
            "invalid_capacity",
        ]:
            # Penalty for invalid action parameters
            reward += self.invalid_action_penalty

        # Add profit-based reward component using configured multiplier
        total_profit = info.get("total_profit", 0.0)
        reward += total_profit * self.profit_multiplier

        # Future: Can add other reward components here, such as:
        # - Diversity bonus for maintaining multiple sectors
        # - Efficiency bonus for low logistic costs
        # - Growth bonus for increasing number of firms
        # - Stability bonus for low death rate

        return reward

    def reset( # type: ignore
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[dict, dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed, options=options)

        self.companies = []
        self.num_firms = 0
        self.current_step = 0

        # Optionally start with some initial companies
        if options and options.get("initial_firms", 0) > 0:
            for _ in range(options["initial_firms"]):
                sector_id = int(self.np_random.integers(0, self.num_sectors))
                # Use configured capital range
                # capital = float(
                #     self.np_random.uniform(
                #         self.initial_capital_min, self.initial_capital_max
                #     )
                # )
                capital = self.initial_capital_max + 1
                while capital <self.initial_capital_min or capital > self.initial_capital_max:
                    capital = self.np_random.beta(1.5,3.5)*self.initial_capital_max
                capital = float(capital)
                

                # Use fixed or random locations based on ablation setting
                if self.fixed_locations:
                    # Fixed grid locations
                    idx = len(self.companies)
                    grid_size = int(np.sqrt(self.max_company)) + 1
                    x = (idx % grid_size) * (self.size / grid_size)
                    y = (idx // grid_size) * (self.size / grid_size)
                    location = (float(x), float(y))
                else:
                    # Random locations
                    location = (
                        float(self.np_random.uniform(0, self.size)),
                        float(self.np_random.uniform(0, self.size)),
                    )

                self.companies.append(
                    Company(
                        capital,
                        sector_id,
                        location,
                        op_cost_rate=self.op_cost_rate,
                        logistic_cost_rate=self.logistic_cost_rate,
                        revenue_rate=self.revenue_rate,
                        min_distance_epsilon=self.min_distance_epsilon,
                        production_capacity_ratio=self.production_capacity_ratio,
                        purchase_budget_ratio=self.purchase_budget_ratio,
                    )
                )
                self.num_firms += 1

        # Build supply chain network after creating companies
        if self.enable_products:
            self._build_supply_chain_network()

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment based on a single continuous action.

        Args:
            action: Numpy array with 4 continuous values:
                [0]: action_type (0.0-1.0, <0.5=invest, >=0.5=create)
                [1]: amount (0.0-1.0, normalized)
                [2]: x coordinate (-20.0 to 20.0 km)
                [3]: y coordinate (-20.0 to 20.0 km)

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        # Track action results for reward calculation
        action_result = "unknown"
        investment_amount = 0.0
        num_valid_actions = 0
        num_invalid_actions = 0

        # Decode action (ensure it's a numpy array)
        action = np.asarray(action, dtype=np.float32)
        
        # Determine action type (invest or create)
        action_type_value = float(action[0])
        is_create = action_type_value >= 0.5
        
        # Get normalized amount (0-1) and denormalize it
        amount_normalized = float(np.clip(action[1], 0.0, 1.0))
        
        # Get coordinates (clamp to valid range)
        x_coord = float(np.clip(action[2], self.map_min, self.map_max))
        y_coord = float(np.clip(action[3], self.map_min, self.map_max))

        if not is_create:
            # Investment action
            # Denormalize amount to investment range
            amt = self.investment_min + amount_normalized * (self.investment_max - self.investment_min)
            
            # Find the closest company to (x_coord, y_coord)
            if self.num_firms > 0:
                min_distance = float('inf')
                closest_company_idx = 0
                
                for idx, company in enumerate(self.companies):
                    dist = np.sqrt(
                        (company.location[0] - x_coord) ** 2 +
                        (company.location[1] - y_coord) ** 2
                    )
                    if dist < min_distance:
                        min_distance = dist
                        closest_company_idx = idx
                
                # Invest in the closest company
                self.companies[closest_company_idx].invest(amt)
                action_result = "valid_invest"
                investment_amount = amt
                num_valid_actions += 1
            else:
                # No companies to invest in
                action_result = "invalid_no_firms"
                num_invalid_actions += 1

        else:
            # Create new company
            # Denormalize amount to creation capital range
            init_capital = self.new_company_capital_min + amount_normalized * (
                self.new_company_capital_max - self.new_company_capital_min
            )
            
            # Choose sector randomly
            sector_id = int(self.np_random.integers(0, self.num_sectors))
            
            # Use the specified location (unless fixed_locations is enabled)
            if self.fixed_locations:
                # Use grid location based on current number of companies
                idx = len(self.companies)
                grid_size = int(np.sqrt(self.max_company)) + 1
                x = (idx % grid_size) * (self.size / grid_size)
                y = (idx // grid_size) * (self.size / grid_size)
                location = (float(x), float(y))
            else:
                # Use the action-specified location
                location = (x_coord, y_coord)
            
            # Validate we can create a new company
            if self.num_firms < self.max_company:
                new_company = Company(
                    init_capital,
                    sector_id,
                    location,
                    op_cost_rate=self.op_cost_rate,
                    logistic_cost_rate=self.logistic_cost_rate,
                    revenue_rate=self.revenue_rate,
                    min_distance_epsilon=self.min_distance_epsilon,
                    production_capacity_ratio=self.production_capacity_ratio,
                    purchase_budget_ratio=self.purchase_budget_ratio,
                )
                self.companies.append(new_company)
                self.num_firms += 1
                action_result = "valid_create"
                num_valid_actions += 1

                # Rebuild supply chain network when new company is added
                if self.enable_products:
                    self._build_supply_chain_network()
            else:
                # Cannot create more companies (at capacity)
                action_result = "invalid_capacity"
                num_invalid_actions += 1

        # Simulate supply chain interactions (applies logistic costs based on distance)
        self._simulate_supply_chain()

        # Execute company operations for all firms
        total_profit = 0.0
        total_logistic_cost = 0.0
        for company in self.companies:
            total_logistic_cost += company.logistic_cost
            profit = company.step()
            total_profit += profit

        # Remove companies that fall below death threshold
        # Unless allow_negative_capital is enabled
        num_deaths = 0
        if not self.allow_negative_capital:
            companies_to_remove = []
            for company in self.companies:
                if company.capital < self.death_threshold:
                    companies_to_remove.append(company)
                    num_deaths += 1

            for company in companies_to_remove:
                self.companies.remove(company)
                self.num_firms -= 1

        # Calculate product-related statistics
        if self.enable_products:
            total_inventory = sum(c.product_inventory for c in self.companies)
            total_produced = sum(c.products_produced_this_step for c in self.companies)
            total_sold = sum(c.products_sold_this_step for c in self.companies)
            total_purchased = sum(
                c.products_purchased_this_step for c in self.companies
            )
        else:
            total_inventory = 0.0
            total_produced = 0.0
            total_sold = 0.0
            total_purchased = 0.0

        # Build info dict with action result
        info = {
            "action_result": action_result,
            "action_type": "create" if is_create else "invest",
            "num_valid_actions": num_valid_actions,
            "num_invalid_actions": num_invalid_actions,
            "investment_amount": investment_amount,
            "total_profit": total_profit,
            "total_logistic_cost": total_logistic_cost,
            "num_firms": self.num_firms,
            "num_deaths": num_deaths,
            "total_inventory": total_inventory,
            "total_produced": total_produced,
            "total_sold": total_sold,
            "total_purchased": total_purchased,
        }

        # Calculate total reward using the dedicated method (pass aggregated results)
        total_reward = self._calculate_reward(info)

        # Build observation
        obs = self._get_observation()

        # Termination conditions
        terminated = False
        truncated = (
            self.current_step >= self.max_episode_steps
        )  # Use configured max episode length

        # Periodic visualization
        if self.visualize_every_n_steps > 0 and self.current_step % self.visualize_every_n_steps == 0:
            self.visualize_step()

        return obs, total_reward, terminated, truncated, info

    def visualize_step(self, step_label: Optional[str] = None) -> None:
        """
        Visualize the current state of the environment.
        
        Args:
            step_label: Optional label for the visualization. Defaults to current step number.
        """
        # Import visualization functions here to avoid circular imports
        from utils.visualize import create_dashboard
        import matplotlib.pyplot as plt
        
        if step_label is None:
            step_label = f"step_{self.current_step:06d}"
        
        # Use the create_dashboard function which expects an IndustryEnv object
        create_dashboard(
            self,
            figsize=(self.figsize_width, self.figsize_height)
        )
        
        # Save figure if enabled
        if self.save_plots:
            # Ensure the visualization directory exists
            Path(self.visualization_dir).mkdir(parents=True, exist_ok=True)
            filepath = Path(self.visualization_dir) / f"{step_label}.{self.plot_format}"
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        
        # Show figure if enabled
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def save_environment(self, filepath: Optional[str] = None) -> str:
        """
        Save the current state of the environment to a file.
        
        Args:
            filepath: Path to save the environment. If None, generates a timestamped filename.
            
        Returns:
            Path to the saved file.
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"environment_{timestamp}.pkl"
        
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Create state dict with all necessary information
        state = {
            'companies': self.companies,
            'num_firms': self.num_firms,
            'current_step': self.current_step,
            'env_config': self.env_config,
            'size': self.size,
            'max_company': self.max_company,
            'num_sectors': self.num_sectors,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        return filepath
