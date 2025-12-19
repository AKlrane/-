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
            self.fixed_investment_amount = getattr(env_config, "fixed_investment_amount", 50000.0)
            self.new_company_capital_min = env_config.new_company_capital_min
            self.new_company_capital_max = env_config.new_company_capital_max
            self.death_threshold = env_config.death_threshold
            self.fixed_cost_per_step = getattr(env_config, "fixed_cost_per_step", -5.0)

            # Supply chain
            self.enable_supply_chain = env_config.enable_supply_chain
            self.trade_volume_fraction = env_config.trade_volume_fraction
            self.revenue_rate = env_config.revenue_rate
            self.min_distance_epsilon = env_config.min_distance_epsilon
            self.nearest_suppliers_count = getattr(env_config, "nearest_suppliers_count", 5)

            # Logistics
            self.disable_logistic_costs = env_config.disable_logistic_costs
            self.tier_logistic_cost_rate = getattr(
                env_config, "tier_logistic_cost_rate", {
                    "Raw": 0.05,
                    "Parts": 0.05,
                    "Electronics": 0.05,
                    "Battery/Motor": 0.05,
                    "OEM": 0.05,
                    "Service": 0.05,
                    "Other": 0.05
                }
            )

            # Management cost
            self.max_capital = getattr(env_config, "max_capital", 100000000.0)
            
            # Pricing - 必须从config读取，没有默认值
            self.tier_prices = getattr(env_config, "tier_prices", {})
            self.tier_cogs = getattr(env_config, "tier_cogs", {})

            # Rewards
            self.revenue_multiplier = env_config.revenue_multiplier
            self.creation_reward = env_config.creation_reward
            self.invalid_coordinate_penalty = env_config.invalid_coordinate_penalty

            # Product system parameters
            self.enable_products = getattr(env_config, "enable_products", True)
            self.tier_production_ratios = getattr(
                env_config, "tier_production_ratios", {
                    "Raw": 0.5,
                    "Parts": 0.3,
                    "Electronics": 0.3,
                    "Battery/Motor": 0.3,
                    "OEM": 0.2,
                    "Service": 0.1,
                    "Other": 0.1
                }
            )
            self.max_held_capital_rate = getattr(
                env_config, "max_held_capital_rate", {
                    "Raw": 0.3,
                    "Parts": 0.4,
                    "Electronics": 0.4,
                    "Battery/Motor": 0.4,
                    "OEM": 0.3,
                    "Service": 0.2,
                    "Other": 0.3
                }
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
            self.fixed_investment_amount = 50000.0
            self.new_company_capital_min = 1000.0
            self.new_company_capital_max = 1000000.0
            self.death_threshold = 0.0
            self.fixed_cost_per_step = -5.0
            self.enable_supply_chain = True
            self.trade_volume_fraction = 0.01
            self.revenue_rate = 1.0
            self.min_distance_epsilon = 0.1
            self.nearest_suppliers_count = 5
            self.disable_logistic_costs = False
            self.tier_logistic_cost_rate = {
                "Raw": 0.05,
                "Parts": 0.05,
                "Electronics": 0.05,
                "Battery/Motor": 0.05,
                "OEM": 0.05,
                "Service": 0.05,
                "Other": 0.05
            }
            self.max_capital = 100000000.0
            # 使用默认配置时，pricing必须从config读取，这里使用空字典
            self.tier_prices = {}
            self.tier_cogs = {}
            self.revenue_multiplier = 0.001
            self.creation_reward = 5.0
            self.invalid_coordinate_penalty = -10.0

            # Product system defaults
            self.enable_products = True
            
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

        # Calculate map boundaries based on size (centered at origin)
        # size = edge length, so coordinates range from -size/2 to +size/2
        self.map_min = -self.size / 2.0
        self.map_max = self.size / 2.0
        
        # Simplified continuous action space for single action per step
        # Box action space with 4 continuous components:
        #   [0]: action_type (0.0-1.0, <0.5=invest, >=0.5=create)
        #   [1]: x coordinate (continuous value in range [map_min, map_max])
        #   [2]: y coordinate (continuous value in range [map_min, map_max])
        #   [3]: tier (0.0-1.0, maps to [0,7] -> ceil -> 1-7 -> sector_id 0-6)
        # Note: Investment amount is fixed (configured via fixed_investment_amount)
        #       Creation capital is random within [new_company_capital_min, new_company_capital_max]
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, self.map_min, self.map_min, 0.0], dtype=np.float32),
            high=np.array([1.0, self.map_max, self.map_max, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.current_step = 0

    def _build_supply_chain_network(self):
        """
        Build supply chain relationships between companies based on their tiers.
        Companies can only buy from the immediately preceding tier.
        
        Supply chain flow:
        - Tier 0 (Raw) → Tier 1 (Parts/Electronics/Battery)
        - Tier 1 (Parts/Electronics/Battery) → Tier 2 (OEM)
        - Tier 2 (OEM) → Tier 3 (Service)
        """
        # Clear existing relationships
        for company in self.companies:
            company.suppliers = []
            company.customers = []

        # Build relationships based on tier adjacency
        for company in self.companies:
            # Find suppliers from the immediately preceding tier only
            for other in self.companies:
                if other == company:
                    continue

                # Check for valid tier-based supply relationships
                # Allow purchasing only from immediately preceding tier
                if company.tier > 0 and other.tier == company.tier - 1:
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

        Respects enable_supply_chain and enable_products flags.
        """
        # Skip if supply chain or products are disabled
        if not self.enable_supply_chain or not self.enable_products:
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

        # Step 2: Demand collection and supply allocation (buyer-driven model)
        # Phase 2a: All buyers calculate needs and place orders with nearest K suppliers
        max_tier = max(c.tier for c in self.companies) if self.companies else 0
        from .sector import sector_relations
        from collections import defaultdict
        K = self.nearest_suppliers_count
        
        # Collect all orders for each supplier: {supplier: [(customer, distance, units_requested)]}
        supplier_orders = defaultdict(list)
        
        for customer in self.companies:
            if customer.tier == 0 or not customer.suppliers:
                continue  # Tier 0 doesn't buy
            
            # INVENTORY LIMIT CHECK: Skip purchasing if inventory value > threshold% of capital
            # Calculate current product inventory value (not raw materials, but finished products)
            inventory_value = customer.product_inventory * customer.revenue_rate
            inventory_ratio = inventory_value / max(customer.capital, 1e-8)
            
            # Get sector-specific max_held_capital_rate threshold
            cust_sector_name = sector_relations[customer.sector_id].name
            max_inventory_ratio = self.max_held_capital_rate.get(cust_sector_name, 0.3)
            
            if inventory_ratio > max_inventory_ratio:
                # Skip purchasing this round if inventory is too high
                continue
            
            # Calculate total purchase budget for this customer
            purchase_budget = customer.get_max_purchase_budget()
            if purchase_budget <= 0:
                continue
            
            cust_name = sector_relations[customer.sector_id].name
            
            # NEW DECOUPLED LOGIC: OEM now buys from nearest K suppliers regardless of type
            if cust_name == "OEM":
                # Filter for relevant suppliers (Parts, Electronics, Battery/Motor)
                relevant_suppliers = [s for s in customer.suppliers 
                                     if sector_relations[s.sector_id].name in ("Parts", "Electronics", "Battery/Motor")]
                
                if relevant_suppliers:
                    # Find nearest K suppliers from all component types
                    nearest_suppliers = sorted(relevant_suppliers, key=lambda s: customer.distance_to(s))[:K]
                    budget_per_supplier = purchase_budget / len(nearest_suppliers)
                    
                    for supplier in nearest_suppliers:
                        unit_price = supplier.revenue_rate
                        units_requested = budget_per_supplier / max(unit_price, 1e-8)
                        if units_requested > 0:
                            dist = customer.distance_to(supplier)
                            supplier_orders[supplier].append((customer, dist, units_requested))
            else:
                # For non-OEM customers: simple nearest K supplier distribution
                nearest_suppliers = sorted(customer.suppliers, key=lambda s: customer.distance_to(s))[:K]
                if not nearest_suppliers:
                    continue
                
                budget_per_supplier = purchase_budget / len(nearest_suppliers)
                
                for supplier in nearest_suppliers:
                    unit_price = supplier.revenue_rate
                    units_requested = budget_per_supplier / max(unit_price, 1e-8)
                    if units_requested > 0:
                        dist = customer.distance_to(supplier)
                        supplier_orders[supplier].append((customer, dist, units_requested))
        
        # Phase 2b: Each supplier fulfills orders from nearest to farthest
        for supplier in self.companies:
            if supplier.product_inventory <= 0:
                continue
            
            orders = supplier_orders[supplier]
            if not orders:
                continue
            
            # Sort orders by distance (nearest first)
            orders.sort(key=lambda x: x[1])
            
            # Fulfill orders sequentially
            for idx, (customer, dist, units_requested) in enumerate(orders):
                if supplier.product_inventory <= 0:
                    break
                
                unit_price = supplier.revenue_rate
                
                # Determine units to sell based on inventory only
                # Note: units_requested already respects purchase_budget from Phase 2a
                units_to_sell = min(units_requested, supplier.product_inventory)
                
                if units_to_sell <= 0:
                    continue
                
                # Perform transaction (revenue/cost recognition, not cash payment)
                cost = units_to_sell * unit_price
                # Note: customer does NOT pay cash now - cost will be recognized through COGS when they sell
                # But we limit their purchasing based on capital (credit limit)
                supplier.add_revenue(units_to_sell)
                supplier.add_cogs(units_to_sell)
                supplier.product_inventory -= units_to_sell
                supplier.products_sold_this_step += units_to_sell
                
                # Add logistic cost to customer (paid in cash immediately)
                if not self.disable_logistic_costs and hasattr(supplier, "logistic_cost_rate"):
                    logistic_cost = supplier.logistic_cost_rate * unit_price * units_to_sell * max(dist, supplier.min_distance_epsilon)
                    customer.capital -= logistic_cost  # Pay logistics cost in cash
                
                # Track input cost per unit for customer's COGS calculation
                sup_name = sector_relations[supplier.sector_id].name
                cust_name = sector_relations[customer.sector_id].name
                unit_cost = cost / max(units_to_sell, 1e-8)
                
                # Store cost tracking by input type
                if sup_name == "Raw":
                    if 'raw' not in customer.input_cost_per_unit:
                        customer.input_cost_per_unit['raw'] = unit_cost
                    else:
                        # Average the cost if buying from multiple suppliers
                        customer.input_cost_per_unit['raw'] = (customer.input_cost_per_unit['raw'] + unit_cost) / 2
                elif sup_name == "Parts":
                    customer.input_cost_per_unit['parts'] = unit_cost
                elif sup_name == "Electronics":
                    customer.input_cost_per_unit['elec'] = unit_cost
                elif sup_name == "Battery/Motor":
                    customer.input_cost_per_unit['batt'] = unit_cost
                elif sup_name == "OEM":
                    customer.input_cost_per_unit['oem'] = unit_cost
                
                # Receive goods into appropriate inventory
                if cust_name in ("Parts", "Electronics", "Battery/Motor") and sup_name == "Raw":
                    customer.raw_inventory += units_to_sell
                elif cust_name == "OEM":
                    if sup_name == "Parts":
                        customer.parts_inventory += units_to_sell
                    elif sup_name == "Electronics":
                        customer.electronics_inventory += units_to_sell
                    elif sup_name == "Battery/Motor":
                        customer.battery_inventory += units_to_sell
                    else:
                        customer.product_inventory += units_to_sell
                elif cust_name == "Service" and sup_name == "OEM":
                    customer.oem_inventory += units_to_sell
                else:
                    customer.product_inventory += units_to_sell
                
                # Track purchase quantity for customer
                customer.products_purchased_this_step += units_to_sell

        # Step 3: After materials flow, run production for all companies
        for company in self.companies:
            company.produce_products()

        # Step 4: Service companies (terminal tier) automatically sell all inventory
        # This represents direct market sales/services that don't require customers
        # NOTE: This must happen BEFORE company.step() to properly account for revenue
        for company in self.companies:
            sector_name = sector_relations[company.sector_id].name
            if sector_name == "Service" and company.product_inventory > 0:
                # Service sells all inventory (direct to market)
                units_sold = company.product_inventory
                revenue = units_sold * company.revenue_rate
                company.revenue += revenue  # Add revenue directly (will be processed in step())
                company.add_cogs(units_sold)  # Apply COGS for the sold units
                company.product_inventory -= units_sold
                company.products_sold_this_step += units_sold

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

        Reward formula:
        reward = log10(net_growth + 1) if net_growth > 0 (profitable, scaled by magnitude)
               = -1.0 if net_growth < 0 (loss, fixed penalty)
               = 0.0 if net_growth == 0 (break-even)
               + creation_reward (if created)
               + invalid_coordinate_penalty (if coordinates out of bounds)

        Args:
            info: Information dict containing:
                - capital_growth: Total capital change in this step
                - investment_amount: Amount invested (if action is invest)
                - action_result: Result of the action
                - coordinates_out_of_bounds: Whether coordinates were out of map range

        Returns:
            Total reward
        """
        reward = 0.0

        # 1. System revenue reward: Log-scaled for profit, fixed penalty for loss
        capital_growth = info.get("capital_growth", 0.0)
        investment_amount = info.get("investment_amount", 0.0)
        net_growth = capital_growth - investment_amount
        if net_growth > 0:
            # 正利润：使用log10压缩，保留规模信息
            reward += np.log10(net_growth + 1.0)
        elif net_growth < 0:
            # 负利润：统一为-1.0，简化处理
            reward += -1.0
        # net_growth == 0 时 reward += 0，保持不变

        # 2. Creation reward: Fixed bonus for creating new companies
        action_result = info.get("action_result", "unknown")
        if action_result == "valid_create":
            reward += self.creation_reward

        # 3. Coordinate penalty: Penalize out-of-bounds coordinates
        coordinates_out_of_bounds = info.get("coordinates_out_of_bounds", False)
        if coordinates_out_of_bounds:
            reward += self.invalid_coordinate_penalty

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
                

                # Random locations (centered at origin)
                location = (
                    float(self.np_random.uniform(self.map_min, self.map_max)),
                    float(self.np_random.uniform(self.map_min, self.map_max)),
                )

                # Get sector-specific production capacity ratio
                sector_name = sector_relations[sector_id].name
                sector_production_ratio = self.tier_production_ratios.get(
                    sector_name, 0.1  # Default fallback if sector not in tier_production_ratios
                )
                sector_logistic_rate = self.tier_logistic_cost_rate.get(
                    sector_name, 0.05  # Default fallback
                )
                
                self.companies.append(
                    Company(
                        capital,
                        sector_id,
                        location,
                        op_cost_rate=self.op_cost_rate,
                        logistic_cost_rate=sector_logistic_rate,
                        revenue_rate=self.revenue_rate,
                        min_distance_epsilon=self.min_distance_epsilon,
                        production_capacity_ratio=sector_production_ratio,
                        fixed_income=self.fixed_cost_per_step,
                        tier_prices=self.tier_prices,
                        tier_cogs=self.tier_cogs,
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
                [1]: x coordinate (map_min to map_max, i.e., -size/2 to +size/2)
                [2]: y coordinate (map_min to map_max, i.e., -size/2 to +size/2)
                [3]: tier (0.0-1.0, maps to sector_id 0-6)

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        # Record total capital before action
        total_capital_before = sum(c.capital for c in self.companies)

        # Track action results for reward calculation
        action_result = "unknown"
        investment_amount = 0.0
        num_valid_actions = 0
        num_invalid_actions = 0
        coordinates_out_of_bounds = False

        # Decode action (ensure it's a numpy array)
        action = np.asarray(action, dtype=np.float32)
        
        # Determine action type (invest or create)
        action_type_value = float(action[0])
        is_create = action_type_value >= 0.5
        
        # Get coordinates and check if they are out of bounds
        x_coord_raw = float(action[1])
        y_coord_raw = float(action[2])
        
        # Check if coordinates are out of bounds
        if (x_coord_raw < self.map_min or x_coord_raw > self.map_max or
            y_coord_raw < self.map_min or y_coord_raw > self.map_max):
            coordinates_out_of_bounds = True
        
        # Clamp coordinates to valid range for execution
        x_coord = float(np.clip(x_coord_raw, self.map_min, self.map_max))
        y_coord = float(np.clip(y_coord_raw, self.map_min, self.map_max))
        
        # Get tier and map to sector_id
        # tier [0,1] -> [0,7] -> ceil -> 1-7 -> sector_id 0-6
        tier_normalized = float(np.clip(action[3], 0.0, 1.0))
        tier_value = tier_normalized * 7.0  # Maps to [0, 7]
        tier_ceiled = int(np.ceil(tier_value))  # Maps to [0, 7], but 0 becomes 1
        # Clamp to 1-7 and convert to sector_id 0-6
        tier_ceiled = max(1, min(7, tier_ceiled))
        target_sector_id = tier_ceiled - 1  # Maps 1-7 to 0-6

        if not is_create:
            # Investment action - use fixed investment amount
            amt = self.fixed_investment_amount
            
            # Find the closest company to (x_coord, y_coord) with matching sector_id
            if self.num_firms > 0:
                min_distance = float('inf')
                closest_company_idx = None
                
                for idx, company in enumerate(self.companies):
                    # Only consider companies with matching sector_id
                    if company.sector_id == target_sector_id:
                        dist = np.sqrt(
                            (company.location[0] - x_coord) ** 2 +
                            (company.location[1] - y_coord) ** 2
                        )
                        if dist < min_distance:
                            min_distance = dist
                            closest_company_idx = idx
                
                # Invest in the closest company with matching tier
                if closest_company_idx is not None:
                    self.companies[closest_company_idx].invest(amt)
                    action_result = "valid_invest"
                    investment_amount = amt
                    num_valid_actions += 1
                else:
                    # No companies with matching tier - do nothing (no penalty)
                    action_result = "no_action"
            else:
                # No companies to invest in - do nothing (no penalty)
                action_result = "no_action"

        else:
            # Create new company
            # Use random initial capital within configured range
            init_capital = float(self.np_random.uniform(
                self.new_company_capital_min,
                self.new_company_capital_max
            ))
            
            # Use the specified sector_id from tier dimension
            sector_id = target_sector_id
            
            # Use the action-specified location
            location = (x_coord, y_coord)
            
            # Validate we can create a new company
            if self.num_firms < self.max_company:
                # Get sector-specific production capacity ratio and logistic cost rate
                sector_name = sector_relations[sector_id].name
                sector_production_ratio = self.tier_production_ratios.get(
                    sector_name, 0.1  # Default fallback if sector not in tier_production_ratios
                )
                sector_logistic_rate = self.tier_logistic_cost_rate.get(
                    sector_name, 0.05  # Default fallback
                )
                
                new_company = Company(
                    init_capital,
                    sector_id,
                    location,
                    op_cost_rate=self.op_cost_rate,
                    logistic_cost_rate=sector_logistic_rate,
                    revenue_rate=self.revenue_rate,
                    min_distance_epsilon=self.min_distance_epsilon,
                    production_capacity_ratio=sector_production_ratio,
                    fixed_income=self.fixed_cost_per_step,
                    tier_prices=self.tier_prices,
                    tier_cogs=self.tier_cogs,
                )
                self.companies.append(new_company)
                self.num_firms += 1
                action_result = "valid_create"
                num_valid_actions += 1
                # 修复：将新公司的初始资本计入investment_amount，作为成本扣除
                # 这样在计算net_growth时会正确扣除创建新公司的成本
                investment_amount = init_capital

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
            profit = company.step(max_capital=self.max_capital)
            total_profit += profit

        # Remove companies that fall below death threshold
        num_deaths = 0
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

        # Calculate total capital after action and company operations
        total_capital_after = sum(c.capital for c in self.companies)
        capital_growth = total_capital_after - total_capital_before

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
            "total_capital_before": total_capital_before,
            "total_capital_after": total_capital_after,
            "capital_growth": capital_growth,
            "coordinates_out_of_bounds": coordinates_out_of_bounds,
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
