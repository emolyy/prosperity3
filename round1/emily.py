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

    def resin_strategy(self, order_depth: OrderDepth, position: int) -> List[Order]:
        """Stable value strategy with tight spreads"""
        orders = []
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        
        # Fixed fair value for stable asset
        fair_value = 10000
        spread = 1
        
        if best_ask and best_ask <= fair_value - spread:
            quantity = min(-order_depth.sell_orders[best_ask], 
                          self.position_limits["RAINFOREST_RESIN"] - position)
            orders.append(Order("RAINFOREST_RESIN", best_ask, quantity))
            
        if best_bid and best_bid >= fair_value + spread:
            quantity = min(order_depth.buy_orders[best_bid], 
                          self.position_limits["RAINFOREST_RESIN"] + position)
            orders.append(Order("RAINFOREST_RESIN", best_bid, -quantity))
            
        return orders

    def kelp_strategy(self, order_depth: OrderDepth, position: int) -> List[Order]:
        """Volatility strategy with dynamic pricing"""
        orders = []
        # Calculate VWAP for dynamic pricing
        if self.kelp_prices:
            vwap = int(sum(self.kelp_prices[-50:])/len(self.kelp_prices[-50:]))
        else:
            vwap = 10000  # Default if no history
            
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        
        spread = 3 # Wider spread for volatile asset
        
        if best_ask and best_ask <= vwap - spread:
            quantity = min(-order_depth.sell_orders[best_ask], 
                          self.position_limits["KELP"] - position)
            orders.append(Order("KELP", best_ask, quantity))
            
        if best_bid and best_bid >= vwap + spread:
            quantity = min(order_depth.buy_orders[best_bid], 
                          self.position_limits["KELP"] + position)
            orders.append(Order("KELP", best_bid, -quantity))
            
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
        
        # Process each product
        for product in ["RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
            if product not in state.order_depths:
                continue
                
            position = state.position.get(product, 0)
            order_depth = state.order_depths[product]
            
            if product == "RAINFOREST_RESIN":
                orders = self.resin_strategy(order_depth, position)
            elif product == "KELP":
                orders = self.kelp_strategy(order_depth, position)
            elif product == "SQUID_INK":
                orders = self.squid_ink_strategy(order_depth, position)
                
            if orders:
                result[product] = orders
                
        # Maintain state between iterations
        trader_data = json.dumps({
            "kelp_prices": self.kelp_prices,
            "squid_ink_prices": self.squid_ink_prices
        })
        
        logger.flush(state, result, 0, trader_data)
        return result, 0, trader_data