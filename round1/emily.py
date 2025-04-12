from typing import Dict, List, Any
import json
import statistics

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class Trader:
    def __init__(self):
        self.position_limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50
        }
        
        # Price history storage
        self.resin_prices = []
        self.kelp_prices = []
        self.squid_ink_prices = []
        
        # Squid Ink pattern detection
        self.ink_pattern = []
        self.ink_window = 20

    def resin_strategy(self, order_depth: OrderDepth, fair_value: int, width: int,
                     position: int, position_limit: int) -> List[Order]:
        """Generate orders for Rainforest Resin."""
        orders = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Check if there are buy and sell orders in the order depth
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return orders

        # Get best prices from the order book
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        # Find prices above/below fair value
        above_fair = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        below_fair = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        best_above_fair = min(above_fair) if len(above_fair) > 0 else fair_value + 2
        best_below_fair = max(below_fair) if len(below_fair) > 0 else fair_value - 2

        # Take opportunities when price is favorable
        if best_ask < fair_value - width:
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            quantity = min(best_ask_amount, position_limit - position)
            if quantity > 0:
                orders.append(Order("RAINFOREST_RESIN", best_ask, quantity))
                buy_order_volume += quantity

        if best_bid > fair_value + width:
            best_bid_amount = order_depth.buy_orders[best_bid]
            quantity = min(best_bid_amount, position_limit + position)
            if quantity > 0:
                orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
                sell_order_volume += quantity

        # Add market-making orders around the fair value
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", best_below_fair + 1, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", best_above_fair - 1, -sell_quantity))

        return orders    
    
    def kelp_strategy(self, order_depth: OrderDepth, position: int) -> List[Order]:
        orders = []
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        
        if not best_ask or not best_bid:  # Simplified check
            return orders
        
        # NEW: Record market price
        current_price = (best_ask + best_bid) // 2
        self.kelp_prices.append(current_price)  # Critical addition

        recent_prices = self.kelp_prices[-20:] if len(self.kelp_prices) >= 20 else self.kelp_prices
        
        # Calculate volatility (simplified)
        price_range = max(recent_prices) - min(recent_prices) if recent_prices else 0
        volatility = price_range / current_price if current_price != 0 else 0
        
        # Dynamic parameters
        spread = max(2, min(5, int(volatility * current_price)))  # 2-5 price units spread
        position_size = int(self.position_limits["KELP"] * 0.1)  # 10% of max position per trade
        
        # Trading logic - always try to make markets
        # Buy when price drops below average
        if best_ask < current_price - spread:
            buy_qty = min(position_size, self.position_limits["KELP"] - position)
            if buy_qty > 0:
                orders.append(Order("KELP", best_ask, buy_qty))
        
        # Sell when price rises above average
        if best_bid > current_price + spread:
            sell_qty = min(position_size, self.position_limits["KELP"] + position)
            if sell_qty > 0:
                orders.append(Order("KELP", best_bid, -sell_qty))
        
        # Always maintain basic market making
        if not orders:  # If no opportunities, provide liquidity
            if position < self.position_limits["KELP"]:
                orders.append(Order("KELP", current_price - spread//2, position_size))
            if position > -self.position_limits["KELP"]:
                orders.append(Order("KELP", current_price + spread//2, -position_size))
        
        return orders

    def squid_ink_strategy(self, order_depth: OrderDepth, position: int) -> List[Order]:
        """Pattern detection strategy with momentum"""
        orders = []
        
        # Check for valid order book presence
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders  # Skip strategy if missing either side of order book
        
        # Calculate midpoint price safely
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        current_price = (best_ask + best_bid) // 2
        
        # Store price history
        self.squid_ink_prices.append(current_price)
        
        # Pattern detection logic
        if len(self.squid_ink_prices) >= self.ink_window:
            window = self.squid_ink_prices[-self.ink_window:]
            avg = sum(window) / len(window)
            std = statistics.stdev(window)
            
            # Bollinger Bands strategy
            upper_band = avg + 2*std
            lower_band = avg - 2*std
            
            if current_price < lower_band:
                quantity = min(10, self.position_limits["SQUID_INK"] - position)
                orders.append(Order("SQUID_INK", best_ask, quantity))
                
            elif current_price > upper_band:
                quantity = min(10, self.position_limits["SQUID_INK"] + position)
                orders.append(Order("SQUID_INK", best_bid, -quantity))
                
        return orders

    def run(self, state: TradingState):
        result = {}

        for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
            if product not in state.order_depths:
                continue

            position = state.position.get(product, 0)
            order_depth = state.order_depths[product]
            orders = []  # Initialize here

            if product == "RAINFOREST_RESIN":
                position_limit = self.position_limits["RAINFOREST_RESIN"]
                orders = self.resin_strategy(
                    order_depth,
                    10000,
                    1,
                    position,
                    position_limit
                )

            elif product == "KELP":
                orders = self.kelp_strategy(order_depth, position)

            elif product == "SQUID_INK":
                orders = self.squid_ink_strategy(order_depth, position)

            if orders:
                result[product] = orders  # Fixed syntax

        trader_data = json.dumps({
            "kelp_prices": self.kelp_prices,
            "squid_ink_prices": self.squid_ink_prices
        })

        logger.flush(state, result, 0, trader_data)
        return result, 0, trader_data