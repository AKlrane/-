"""
Company module for industry simulation.
Defines the Company class with production, purchasing, and supply chain capabilities.
"""

import numpy as np
from typing import Tuple, List, TYPE_CHECKING

from .sector import SECTOR_TIERS

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
        self.revenue_rate = revenue_rate  # Multiplier for converting orders to revenue
        self.min_distance_epsilon = (
            min_distance_epsilon  # Minimum distance to prevent division by zero
        )
        self.revenue = 0.0
        self.orders = 0
        self.logistic_cost = 0.0  # Track accumulated logistic costs

        # Product system
        self.production_capacity_ratio = (
            production_capacity_ratio  # Max production = capital * ratio
        )
        self.purchase_budget_ratio = (
            purchase_budget_ratio  # Max purchase = capital * ratio
        )
        self.product_inventory = 0.0  # Amount of product in inventory
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
        Uses inverse square law: cost = k * volume / distance^2

        Args:
            other: The destination company
            trade_volume: Volume of goods to transport (default 1.0)

        Returns:
            Logistic cost based on distance and volume
        """
        distance = self.distance_to(other)
        # Prevent division by zero for co-located companies
        # Use configured epsilon

        # Inverse square law: cost increases with volume, decreases with square of distance
        # logistic_cost_rate controls the magnitude
        cost = self.logistic_cost_rate * trade_volume * distance

        return cost

    def step(self) -> float:
        """Execute one time step for the company. Returns profit."""
        op_cost = self.op_cost_rate * self.capital
        # Total cost includes operating cost and accumulated logistic costs
        total_cost = op_cost + self.logistic_cost
        profit = self.revenue - total_cost
        self.capital += profit

        # Reset for next step
        self.revenue = 0.0
        self.logistic_cost = 0.0

        return profit

    def invest(self, amount: float):
        """Add capital investment to the company."""
        self.capital += amount

    def add_revenue(self, order_amount: float):
        """Add revenue from sales/orders using revenue_rate * order_amount."""
        self.revenue += self.revenue_rate * order_amount
        self.orders += 1

    def add_logistic_cost(self, cost: float):
        """Add logistic cost from transportation/supply chain."""
        self.logistic_cost += cost

    def get_max_production(self) -> float:
        """Calculate maximum production capacity based on capital."""
        return self.capital * self.production_capacity_ratio

    def get_max_purchase_budget(self) -> float:
        """Calculate maximum purchase budget based on capital."""
        return self.capital * self.purchase_budget_ratio

    def produce_products(self) -> float:
        """
        Produce products (only for tier 0 - most upstream companies).
        Production amount based on capital and capacity ratio.

        Returns:
            Amount produced
        """
        if self.tier != 0:
            return 0.0

        # Tier 0 companies produce from scratch based on their capital
        production_amount = self.get_max_production()
        self.product_inventory += production_amount
        self.products_produced_this_step = production_amount

        return production_amount

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

        return fulfilled_amount

    def purchase_from_suppliers(
        self, total_orders_from_customers: float = 0.0, disable_logistic_costs: bool = False
    ) -> float:
        """
        Purchase products from upstream suppliers.
        Purchase amount is limited by:
        1. Capital * purchase_budget_ratio (max budget)
        2. Suppliers' total available inventory

        Args:
            total_orders_from_customers: Total orders this company received (for planning)
            disable_logistic_costs: If True, don't add logistic costs

        Returns:
            Total amount purchased
        """
        if self.tier == 0 or not self.suppliers:
            return 0.0  # Tier 0 doesn't buy, produces instead

        # Calculate purchase budget
        max_budget = self.get_max_purchase_budget()

        # Calculate how much we want to purchase
        # Either to fulfill customer orders + maintain inventory, or use full budget
        desired_purchase = max(total_orders_from_customers, max_budget)
        desired_purchase = min(desired_purchase, max_budget)

        if desired_purchase <= 0:
            return 0.0

        # Distribute purchases across suppliers
        amount_per_supplier = desired_purchase / len(self.suppliers)
        total_purchased = 0.0
        total_cost = 0.0

        for supplier in self.suppliers:
            # Request products from supplier
            # Price is based on the supplier's revenue_rate (simplified pricing model)
            unit_price = supplier.revenue_rate
            requested_units = amount_per_supplier

            # Supplier fulfills order from their inventory
            fulfilled_units = supplier.receive_order(requested_units)

            if fulfilled_units > 0:
                # Calculate cost
                cost = fulfilled_units * unit_price

                # Check if we have capital to pay
                if self.capital >= cost:
                    self.capital -= cost
                    self.product_inventory += fulfilled_units
                    total_purchased += fulfilled_units
                    total_cost += cost

                    # Add logistic cost if enabled
                    if not disable_logistic_costs and hasattr(self, "logistic_cost_rate"):
                        logistic_cost = self.calculate_logistic_cost_to(
                            supplier, fulfilled_units
                        )
                        self.add_logistic_cost(logistic_cost)

        self.products_purchased_this_step = total_purchased

        return total_purchased

    def reset_step_counters(self):
        """Reset per-step tracking counters."""
        self.products_produced_this_step = 0.0
        self.products_sold_this_step = 0.0
        self.products_purchased_this_step = 0.0
