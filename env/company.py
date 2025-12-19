"""
Company module for industry simulation.
Defines the Company class with production, purchasing, and supply chain capabilities.
"""

import numpy as np
from typing import Tuple, List, TYPE_CHECKING

from .sector import SECTOR_TIERS, sector_relations

if TYPE_CHECKING:
    from .company import Company


class Company:
    """Represents a company with capital, operating costs, sector, location, and product inventory."""

    def __init__(
        self,
        capital: float,
        sector_id: int,
        location: Tuple[float, float],
        op_cost_rate: float = 0.05,
        logistic_cost_rate: float = 1.0,
        revenue_rate: float = 1.0,
        min_distance_epsilon: float = 0.1,
        production_capacity_ratio: float = 0.1, # Max production = capital * ratio
        fixed_income: float = -5.0,
        tier_prices: dict = None,
        tier_cogs: dict = None,
    ):
        self.capital = capital
        self.sector_id = sector_id
        self.location = location  # (x, y) coordinates
        
        # Apply sector-specific operating cost multiplier
        from .sector import sector_relations
        sector_multiplier = sector_relations[sector_id].operating_cost_multiplier
        self.op_cost_rate = op_cost_rate * sector_multiplier
        
        self.logistic_cost_rate = (
            logistic_cost_rate  # Controls magnitude of distance-based costs
        )
        # Set unit price by sector/tier mapping from config
        # 必须从config传入，没有硬编码默认值
        if tier_prices is None:
            tier_prices = {}
        if tier_cogs is None:
            tier_cogs = {}
        self.tier_prices = tier_prices
        self.tier_cogs = tier_cogs
        sector_name = sector_relations[sector_id].name
        self.revenue_rate = tier_prices.get(sector_name, revenue_rate)
        self.min_distance_epsilon = (
            min_distance_epsilon  # Minimum distance to prevent division by zero
        )
        self.revenue = 0.0
        self.orders = 0
        self.logistic_cost = 0.0  # Track accumulated logistic costs
        self.cogs_cost = 0.0  # Cost of goods sold accumulated within step
        
        # Per-unit COGS for sectors that have explicit unit cost from config
        self.unit_cogs = tier_cogs.get(sector_name, 0.0)
        
        self.input_cost_per_unit = {}  # Track cost of inputs (for calculating COGS on output)
        self.product_unit_cost = 0.0  # Cost per unit of finished product (calculated during production)

        self.fixed_income = fixed_income  # Fixed cost per step (default -20)

        # Product system
        self.production_capacity_ratio = (
            production_capacity_ratio  # Max production = capital * ratio
        )
        # Finished product inventory (what this company sells)
        # Start with some initial inventory to bootstrap supply chain
        # initial_inventory = capital / unit_price / 10
        sector_name = sector_relations[sector_id].name
        
        # Set initial product_unit_cost based on sector
        # This prevents incorrect COGS calculation for initial inventory
        if sector_name == "Raw":
            self.product_unit_cost = self.unit_cogs  # Use config COGS for Raw
        elif sector_name == "Parts":
            # 所有值必须从config的tier_prices读取，没有硬编码默认值
            raw_price = tier_prices.get("Raw", 0.0) if tier_prices else 0.0
            self.product_unit_cost = 3.0 * raw_price
        elif sector_name == "Electronics":
            raw_price = tier_prices.get("Raw", 0.0) if tier_prices else 0.0
            self.product_unit_cost = 7.0 * raw_price
        elif sector_name == "Battery/Motor":
            raw_price = tier_prices.get("Raw", 0.0) if tier_prices else 0.0
            self.product_unit_cost = 20.0 * raw_price
        elif sector_name == "OEM":
            # Approximate OEM cost based on new decoupled logic
            # Average of three production routes: 20*Parts, 10*Electronics, 4*Battery
            # 所有值必须从config的tier_prices读取，没有硬编码默认值
            if tier_prices:
                cost_from_parts = 20 * tier_prices.get("Parts", 0.0)
                cost_from_elec = 10 * tier_prices.get("Electronics", 0.0)
                cost_from_batt = 4 * tier_prices.get("Battery/Motor", 0.0)
                total_cost = cost_from_parts + cost_from_elec + cost_from_batt
                self.product_unit_cost = total_cost / 3.0 if total_cost > 0 else 0.0
            else:
                self.product_unit_cost = 0.0
        elif sector_name == "Service":
            # Service cost based on OEM
            # 所有值必须从config的tier_prices读取，没有硬编码默认值
            oem_price = tier_prices.get("OEM", 0.0) if tier_prices else 0.0
            self.product_unit_cost = 2.0 * oem_price
        else:
            self.product_unit_cost = self.unit_cogs  # Other sectors use unit_cogs
        
        # Initial inventory starts at 0 (no free bootstrap inventory)
        self.product_inventory = 0.0
        # Input inventories for processing/assembly
        self.raw_inventory = 0.0  # For tier 1 processors (raw materials)
        self.parts_inventory = 0.0  # For OEM
        self.electronics_inventory = 0.0  # For OEM
        self.battery_inventory = 0.0  # For OEM (battery/motor)
        self.oem_inventory = 0.0  # For service
        self.tier = SECTOR_TIERS[sector_id]  # Supply chain tier

        # Upstream suppliers (companies this company buys from)
        self.suppliers: List["Company"] = []

        # Downstream customers (companies that buy from this company)
        self.customers: List["Company"] = []

        # Track production and sales for this step
        self.products_produced_this_step = 0.0
        self.products_sold_this_step = 0.0
        self.products_purchased_this_step = 0.0

    @property
    def x(self) -> float:
        """Get x-coordinate of company location."""
        return self.location[0]

    @property
    def y(self) -> float:
        """Get y-coordinate of company location."""
        return self.location[1]

    def distance_to(self, other: "Company") -> float:
        """Calculate Euclidean distance to another company."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def calculate_logistic_cost_to(
        self, other: "Company", trade_volume: float = 1.0
    ) -> float:
        """
        Calculate logistic cost to transport goods to another company.
        New definition: cost = rate * unit_price * volume * distance

        Args:
            other: The destination company
            trade_volume: Volume of goods to transport (units)

        Returns:
            Logistic cost based on distance, unit price and volume
        """
        distance = self.distance_to(other)
        unit_price = self.revenue_rate
        cost = self.logistic_cost_rate * unit_price * trade_volume * max(distance, self.min_distance_epsilon)

        return cost

    def step(self, max_capital: float = 100000000.0) -> float:
        """
        Execute one time step for the company. Returns profit.
        
        Args:
            max_capital: Maximum system capital threshold for calculating management cost
        """
        import math
        
        op_cost = self.op_cost_rate * self.capital
        
        # Polynomial management cost for stronger negative feedback
        # Cost scales with (capital / max_capital)^0.5
        # When capital = max_capital, management_cost = max_capital * 0.001
        capital_ratio = max(self.capital, 0.0) / max(max_capital, 1.0)
        management_cost = max(self.capital, 0.0) * 0.001 * (capital_ratio ** 0.5)
        
        # Total cost includes operating cost, management cost, and COGS
        # Note: Logistic costs are paid in cash immediately, not included here
        total_cost = op_cost + management_cost + self.cogs_cost
        profit = self.revenue - total_cost + self.fixed_income
        self.capital += profit

        # Reset for next step
        self.revenue = 0.0
        self.logistic_cost = 0.0  # Reset tracking variable (already paid in cash)
        self.cogs_cost = 0.0

        return profit

    def invest(self, amount: float):
        """Add capital investment to the company."""
        self.capital += amount

    def add_revenue(self, order_amount: float):
        """Add revenue from sales/orders using revenue_rate * order_amount."""
        self.revenue += self.revenue_rate * order_amount
        self.orders += 1

    def add_cogs(self, units: float):
        """
        Accumulate cost of goods sold for sold units.
        Uses product_unit_cost (calculated during production) for processed goods,
        or unit_cogs (from config) for Raw/Other.
        """
        if units > 0.0:
            # Use product_unit_cost if available (from production), otherwise unit_cogs (Raw/Other)
            cost_per_unit = self.product_unit_cost if self.product_unit_cost > 0.0 else self.unit_cogs
            self.cogs_cost += cost_per_unit * units

    def add_logistic_cost(self, cost: float):
        """Add logistic cost from transportation/supply chain."""
        self.logistic_cost += cost

    def get_max_production(self) -> float:
        """Calculate maximum production capacity based on capital."""
        return self.capital * self.production_capacity_ratio

    def get_max_purchase_budget(self) -> float:
        """
        Calculate maximum purchase budget based on capital.
        Uses production_capacity_ratio so purchase capacity matches production capacity.
        """
        return self.capital * self.production_capacity_ratio

    def produce_products(self) -> float:
        """
        Produce products by converting inputs to outputs.
        - Tier 0 (Raw): produce raw directly based on capacity.
        - Tier 1 processors (Parts/Electronics/Battery): convert raw to product using ratios.
        - Tier 2 (OEM): assemble using parts/electronics/battery with specific recipe.
        - Tier 3 (Service): consume OEM units to produce service units.
        
        Also calculates product_unit_cost for accurate COGS calculation.

        Returns:
            Amount of finished products produced this step
        """
        produced = 0.0
        if self.tier == 0:
            # Produce raw materials directly
            production_amount = self.get_max_production()
            self.product_inventory += production_amount
            produced = production_amount
            # Raw products have unit_cogs from config
            self.product_unit_cost = self.unit_cogs
        else:
            # Determine processing based on sector name
            from .sector import sector_relations
            sector_name = sector_relations[self.sector_id].name
            capacity_units = max(0.0, self.get_max_production())

            if sector_name == "Parts":
                # 3 raw -> 1 parts
                possible = int(self.raw_inventory // 3)
                craft = int(min(possible, capacity_units))
                if craft > 0:
                    self.raw_inventory -= 3 * craft
                    self.product_inventory += craft
                    produced = float(craft)
                    # Cost = 3 × raw_unit_cost
                    raw_cost = self.input_cost_per_unit.get('raw', 0.0)
                    self.product_unit_cost = 3 * raw_cost
            elif sector_name == "Electronics":
                # 7 raw -> 1 electronics
                possible = int(self.raw_inventory // 7)
                craft = int(min(possible, capacity_units))
                if craft > 0:
                    self.raw_inventory -= 7 * craft
                    self.product_inventory += craft
                    produced = float(craft)
                    # Cost = 7 × raw_unit_cost
                    raw_cost = self.input_cost_per_unit.get('raw', 0.0)
                    self.product_unit_cost = 7 * raw_cost
            elif sector_name == "Battery/Motor":
                # 20 raw -> 1 battery/motor
                possible = int(self.raw_inventory // 20)
                craft = int(min(possible, capacity_units))
                if craft > 0:
                    self.raw_inventory -= 20 * craft
                    self.product_inventory += craft
                    produced = float(craft)
                    # Cost = 20 × raw_unit_cost
                    raw_cost = self.input_cost_per_unit.get('raw', 0.0)
                    self.product_unit_cost = 20 * raw_cost
            elif sector_name == "OEM":
                # NEW DECOUPLED LOGIC: 20 parts OR 10 electronics OR 4 battery -> 1 OEM
                # Produce as many as possible from each type independently
                total_produced = 0.0
                total_cost = 0.0
                
                # Try to produce from parts: 20 parts -> 1 OEM
                possible_from_parts = int(self.parts_inventory // 20)
                craft_from_parts = int(min(possible_from_parts, capacity_units - total_produced))
                if craft_from_parts > 0:
                    self.parts_inventory -= 20 * craft_from_parts
                    total_produced += craft_from_parts
                    parts_cost = self.input_cost_per_unit.get('parts', 0.0)
                    total_cost += 20 * parts_cost * craft_from_parts
                
                # Try to produce from electronics: 10 electronics -> 1 OEM
                possible_from_elec = int(self.electronics_inventory // 10)
                craft_from_elec = int(min(possible_from_elec, capacity_units - total_produced))
                if craft_from_elec > 0:
                    self.electronics_inventory -= 10 * craft_from_elec
                    total_produced += craft_from_elec
                    elec_cost = self.input_cost_per_unit.get('elec', 0.0)
                    total_cost += 10 * elec_cost * craft_from_elec
                
                # Try to produce from battery: 4 battery -> 1 OEM
                possible_from_batt = int(self.battery_inventory // 4)
                craft_from_batt = int(min(possible_from_batt, capacity_units - total_produced))
                if craft_from_batt > 0:
                    self.battery_inventory -= 4 * craft_from_batt
                    total_produced += craft_from_batt
                    batt_cost = self.input_cost_per_unit.get('batt', 0.0)
                    total_cost += 4 * batt_cost * craft_from_batt
                
                if total_produced > 0:
                    self.product_inventory += total_produced
                    produced = float(total_produced)
                    # Average cost per unit
                    self.product_unit_cost = total_cost / max(total_produced, 1e-8)
            elif sector_name == "Service":
                # 2 OEM -> 1 Service
                possible = int(self.oem_inventory // 2)
                craft = int(min(possible, capacity_units))
                if craft > 0:
                    self.oem_inventory -= 2 * craft
                    self.product_inventory += craft
                    produced = float(craft)
                    # Cost = 2 × OEM_unit_cost
                    oem_cost = self.input_cost_per_unit.get('oem', 0.0)
                    self.product_unit_cost = 2 * oem_cost

        self.products_produced_this_step = produced
        return produced

    def reset_step_counters(self):
        """Reset per-step tracking counters."""
        self.products_produced_this_step = 0.0
        self.products_sold_this_step = 0.0
        self.products_purchased_this_step = 0.0
