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
        purchase_budget_ratio: float = 0.2,
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
        if tier_prices is None:
            tier_prices = {
                "Raw": 10.0,
                "Parts": 36.0,
                "Electronics": 85.0,
                "Battery/Motor": 250.0,
                "OEM": 3000.0,
                "Service": 7000.0,
                "Other": 10.5,
            }
        if tier_cogs is None:
            tier_cogs = {
                "Raw": 8.5,
                "Other": 10.0
            }
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
        self.purchase_budget_ratio = (
            purchase_budget_ratio  # Max purchase = capital * ratio
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
            self.product_unit_cost = 3.0 * tier_prices.get("Raw", 1.0) if tier_prices else 3.0
        elif sector_name == "Electronics":
            self.product_unit_cost = 7.0 * tier_prices.get("Raw", 1.0) if tier_prices else 7.0
        elif sector_name == "Battery/Motor":
            self.product_unit_cost = 20.0 * tier_prices.get("Raw", 1.0) if tier_prices else 20.0
        elif sector_name == "OEM":
            # Approximate OEM cost based on new decoupled logic
            # Average of three production routes: 20*Parts, 10*Electronics, 4*Battery
            if tier_prices:
                cost_from_parts = 20 * tier_prices.get("Parts", 5.0)
                cost_from_elec = 10 * tier_prices.get("Electronics", 12.0)
                cost_from_batt = 4 * tier_prices.get("Battery/Motor", 35.0)
                self.product_unit_cost = (cost_from_parts + cost_from_elec + cost_from_batt) / 3.0
            else:
                self.product_unit_cost = 160.0  # Default estimate
        elif sector_name == "Service":
            # Service cost based on OEM
            self.product_unit_cost = 2.0 * tier_prices.get("OEM", 450.0) if tier_prices else 900.0
        else:
            self.product_unit_cost = self.unit_cogs  # Other sectors use unit_cogs
        
        # Calculate initial inventory: capital / unit_price / 100 (small amount to bootstrap)
        initial_inventory = self.capital / max(self.revenue_rate, 1.0) / 100.0
        self.product_inventory = initial_inventory
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
        
        # Total cost includes operating cost, management cost, and accumulated logistic costs
        total_cost = op_cost + management_cost + self.logistic_cost + self.cogs_cost
        profit = self.revenue - total_cost + self.fixed_income
        self.capital += profit

        # Reset for next step
        self.revenue = 0.0
        self.logistic_cost = 0.0
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
        """Calculate maximum purchase budget based on capital."""
        return self.capital * self.purchase_budget_ratio

    def calculate_raw_needs(self) -> dict:
        """
        Calculate raw material needs for Parts, Electronics, Battery makers.
        Returns dict with keys: 'parts', 'electronics', 'battery' (units needed from Raw)
        """
        from .sector import sector_relations
        sector_name = sector_relations[self.sector_id].name
        capacity = self.get_max_production()
        
        needs = {'parts': 0, 'electronics': 0, 'battery': 0}
        
        if sector_name == "Parts":
            # Can produce up to capacity units, need 3 raw each
            needed = max(0, capacity - self.product_inventory)
            needs['parts'] = int(needed * 3)
        elif sector_name == "Electronics":
            # Can produce up to capacity units, need 7 raw each
            needed = max(0, capacity - self.product_inventory)
            needs['electronics'] = int(needed * 7)
        elif sector_name == "Battery/Motor":
            # Can produce up to capacity units, need 20 raw each
            needed = max(0, capacity - self.product_inventory)
            needs['battery'] = int(needed * 20)
        
        return needs

    def calculate_oem_needs(self) -> dict:
        """
        Calculate component needs for OEM (simplified - no longer used).
        OEM now buys from any nearest suppliers regardless of type.
        """
        return {'total_budget': self.get_max_purchase_budget()}

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

    def receive_order(self, amount: float) -> float:
        """
        Receive an order from a downstream customer.
        Fulfill as much as possible from inventory.

        Args:
            amount: Requested order amount

        Returns:
            Actual amount fulfilled
        """
        # Limit fulfillment to available inventory
        fulfilled_amount = min(amount, self.product_inventory)

        if fulfilled_amount > 0:
            self.product_inventory -= fulfilled_amount
            self.products_sold_this_step += fulfilled_amount
            # Add revenue (using revenue_rate)
            self.add_revenue(fulfilled_amount)
            # Add COGS for sold units (using unified add_cogs method)
            self.add_cogs(fulfilled_amount)

        return fulfilled_amount

    def purchase_from_suppliers(
        self, total_orders_from_customers: float = 0.0, disable_logistic_costs: bool = False
    ) -> float:
        """
        Purchase products from upstream suppliers with smart prioritization.
        
        For Parts/Electronics/Battery: Calculate raw needs based on production capacity,
                                      then distribute budget among them.
        For OEM: First ensure target inventory levels, then use remaining budget.
        
        Args:
            total_orders_from_customers: Total orders this company received (for planning)
            disable_logistic_costs: If True, don't add logistic costs

        Returns:
            Total amount purchased
        """
        if self.tier == 0 or not self.suppliers:
            return 0.0  # Tier 0 doesn't buy, produces instead

        from .sector import sector_relations
        sector_name = sector_relations[self.sector_id].name
        max_budget = self.get_max_purchase_budget()
        
        if max_budget <= 0:
            return 0.0

        # Choose nearest K suppliers (K=5)
        K = 5
        nearest_suppliers = sorted(self.suppliers, key=lambda s: self.distance_to(s))[:K]
        
        total_purchased = 0.0
        
        # CASE 1: OEM - NEW DECOUPLED LOGIC: buy from nearest K suppliers regardless of type
        if sector_name == "OEM":
            # Simply distribute budget evenly among nearest suppliers
            # Don't care if they sell Parts, Electronics, or Battery - buy whatever they have
            per_supplier_budget = max_budget / len(nearest_suppliers)
            
            for supplier in nearest_suppliers:
                sup_sector = sector_relations[supplier.sector_id].name
                # Only buy from Parts, Electronics, or Battery/Motor suppliers
                if sup_sector in ("Parts", "Electronics", "Battery/Motor"):
                    purchased = self._purchase_from_single_supplier(
                        supplier, per_supplier_budget, disable_logistic_costs
                    )
                    total_purchased += purchased
        
        # CASE 2: Parts/Electronics/Battery - calculate raw needs and buy proportionally
        elif sector_name in ("Parts", "Electronics", "Battery/Motor"):
            needs = self.calculate_raw_needs()
            
            # Get the specific raw need for this sector
            if sector_name == "Parts":
                raw_needed = needs['parts']
            elif sector_name == "Electronics":
                raw_needed = needs['electronics']
            else:  # Battery/Motor
                raw_needed = needs['battery']
            
            if raw_needed > 0 and max_budget > 0:
                # Get actual Raw price from tier_prices
                raw_price = self.tier_prices.get("Raw", 1.0)
                raw_cost = raw_needed * raw_price
                allocated_budget = min(raw_cost, max_budget)
                per_supplier_budget = allocated_budget / len(nearest_suppliers)
                
                for supplier in nearest_suppliers:
                    sup_sector = sector_relations[supplier.sector_id].name
                    if sup_sector == "Raw":
                        purchased = self._purchase_from_single_supplier(
                            supplier, per_supplier_budget, disable_logistic_costs
                        )
                        total_purchased += purchased
        
        # CASE 3: Service - buy OEM products
        elif sector_name == "Service":
            per_supplier_budget = max_budget / len(nearest_suppliers)
            for supplier in nearest_suppliers:
                sup_sector = sector_relations[supplier.sector_id].name
                if sup_sector == "OEM":
                    purchased = self._purchase_from_single_supplier(
                        supplier, per_supplier_budget, disable_logistic_costs
                    )
                    total_purchased += purchased
        
        self.products_purchased_this_step = total_purchased
        return total_purchased

    def _purchase_from_single_supplier(
        self, supplier: "Company", budget: float, disable_logistic_costs: bool = False
    ) -> float:
        """
        Helper method to purchase from a single supplier.
        Returns amount purchased.
        """
        if budget <= 0 or supplier.product_inventory <= 0:
            return 0.0
        
        from .sector import sector_relations
        
        unit_price = supplier.revenue_rate
        requested_units = budget / max(unit_price, 1e-8)
        
        # Supplier fulfills order from their inventory
        fulfilled_units = supplier.receive_order(requested_units)
        
        if fulfilled_units <= 0:
            return 0.0
        
        # Calculate cost
        cost = fulfilled_units * unit_price
        
        # Check if we have capital to pay
        if self.capital < cost:
            fulfilled_units = self.capital / max(unit_price, 1e-8)
            cost = fulfilled_units * unit_price
        
        self.capital -= cost
        
        # Track input cost per unit for future COGS calculation
        my_sector = sector_relations[self.sector_id].name
        sup_sector = sector_relations[supplier.sector_id].name
        unit_cost = cost / max(fulfilled_units, 1e-8)
        
        # Store cost tracking by input type
        if sup_sector == "Raw":
            if 'raw' not in self.input_cost_per_unit:
                self.input_cost_per_unit['raw'] = unit_cost
            else:
                # Average the cost if buying from multiple suppliers
                self.input_cost_per_unit['raw'] = (self.input_cost_per_unit['raw'] + unit_cost) / 2
        elif sup_sector == "Parts":
            self.input_cost_per_unit['parts'] = unit_cost
        elif sup_sector == "Electronics":
            self.input_cost_per_unit['elec'] = unit_cost
        elif sup_sector == "Battery/Motor":
            self.input_cost_per_unit['batt'] = unit_cost
        elif sup_sector == "OEM":
            self.input_cost_per_unit['oem'] = unit_cost
        
        # Add to appropriate input inventory
        my_sector = sector_relations[self.sector_id].name
        sup_sector = sector_relations[supplier.sector_id].name
        
        if my_sector in ("Parts", "Electronics", "Battery/Motor") and sup_sector == "Raw":
            self.raw_inventory += fulfilled_units
        elif my_sector == "OEM":
            if sup_sector == "Parts":
                self.parts_inventory += fulfilled_units
            elif sup_sector == "Electronics":
                self.electronics_inventory += fulfilled_units
            elif sup_sector == "Battery/Motor":
                self.battery_inventory += fulfilled_units
            else:
                self.product_inventory += fulfilled_units
        elif my_sector == "Service" and sup_sector == "OEM":
            self.oem_inventory += fulfilled_units
        else:
            self.product_inventory += fulfilled_units
        
        # Add logistic cost if enabled
        if not disable_logistic_costs and hasattr(self, "logistic_cost_rate"):
            logistic_cost = self.calculate_logistic_cost_to(supplier, fulfilled_units)
            self.add_logistic_cost(logistic_cost)
        
        return fulfilled_units

    def reset_step_counters(self):
        """Reset per-step tracking counters."""
        self.products_produced_this_step = 0.0
        self.products_sold_this_step = 0.0
        self.products_purchased_this_step = 0.0
