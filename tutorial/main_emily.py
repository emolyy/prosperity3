import json
from typing import Any, Dict, List
import statistics
import math

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
            "KELP": 50
        }
        
        # Price history storage
        self.resin_prices = []
        self.resin_vwap = []
        self.kelp_prices = []
        self.kelp_vwap = []
        
        # State tracking
        self.day = 0
        
    def resin_orders(self, order_depth: OrderDepth, fair_value: int, width: int, 
                    position: int, position_limit: int) -> List[Order]:
        """Generate orders for Rainforest Resin"""
        orders = []
        
        buy_order_volume = 0
        sell_order_volume = 0
        
        # Find relevant price levels
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return orders
            
        # Get best prices
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        
        # Find best prices above/below fair value
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
                
        # Add position management orders
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "RAINFOREST_RESIN", 
            buy_order_volume, sell_order_volume, fair_value, 1
        )
        
        # Add market making orders
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", best_below_fair + 1, buy_quantity))
            
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", best_above_fair - 1, -sell_quantity))
            
        return orders
        
    def kelp_orders(self, order_depth: OrderDepth, timespan: int, make_width: int, 
                   take_width: int, position: int, position_limit: int) -> List[Order]:
        """Generate orders for Kelp"""
        orders = []
        
        buy_order_volume = 0
        sell_order_volume = 0
        
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return orders
            
        # Get best prices
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        
        # Find MM bid/ask - prices with significant volume
        # In kelp_orders() method:
        filtered_ask = [p for p in order_depth.sell_orders if abs(order_depth.sell_orders[p]) >= 15]
        filtered_bid = [p for p in order_depth.buy_orders if abs(order_depth.buy_orders[p]) >= 15]
        mm_ask = min(filtered_ask) if filtered_ask else best_ask
        mm_bid = max(filtered_bid) if filtered_bid else best_bid
        fair_value = (mm_ask + mm_bid) / 2  # Float precision
            
        # Convert to integer
        fair_value_int = int(round(fair_value))
            
        # Take all favorable orders
        if best_ask <= fair_value_int - take_width:
            ask_amount = -1 * order_depth.sell_orders[best_ask]
            if ask_amount <= 20:  # Limit size for taking liquidity
                quantity = min(ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order("KELP", best_ask, quantity))
                    buy_order_volume += quantity
                    
        if best_bid >= fair_value_int + take_width:
            bid_amount = order_depth.buy_orders[best_bid]
            if bid_amount <= 20:  # Limit size for taking liquidity
                quantity = min(bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order("KELP", best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    
        # Add position management orders
        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "KELP", 
            buy_order_volume, sell_order_volume, fair_value_int, 2
        )
        
        # Find prices for market making
        above_fair = [price for price in order_depth.sell_orders.keys() if price > fair_value_int + 1]
        below_fair = [price for price in order_depth.buy_orders.keys() if price < fair_value_int - 1]
        
        best_above_fair = min(above_fair) if len(above_fair) > 0 else fair_value_int + 2
        best_below_fair = max(below_fair) if len(below_fair) > 0 else fair_value_int - 2
        
        # Add market making orders
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("KELP", best_below_fair + 1, buy_quantity))
            
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("KELP", best_above_fair - 1, -sell_quantity))
            
        return orders
        
    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, 
                            position_limit: int, product: str, buy_order_volume: int, 
                            sell_order_volume: int, fair_value: int, width: int) -> tuple:
        """Add orders to reduce position when needed"""
        position_after_take = position + buy_order_volume - sell_order_volume
        fair = fair_value
        fair_for_bid = fair_value
        fair_for_ask = fair_value
        
        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)
        
        # If we're net long after taking favorable prices, try to reduce position at fair value
        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                    sell_order_volume += abs(sent_quantity)
                
        # If we're net short after taking favorable prices, try to increase position at fair value
        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                    buy_order_volume += abs(sent_quantity)
                    
        return buy_order_volume, sell_order_volume
        
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """Main method called by the exchange."""
        # Initialize result containers
        result = {}
        conversions = 0
        
        # Increment day counter
        self.day += 1
        
        # Set parameters
        resin_fair_value = 10000  # Fixed fair value for stable Resin
        resin_width = 1  # Small width for Resin due to stability

        kelp_make_width = 3.5  # Increased from 2 for better spread capture
        kelp_take_width = 1.5  # Adjusted from 1 for volatility tolerance
        kelp_timespan = 15     # Extended from 10 for smoother average
        
        # Load previous state if available
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
                self.resin_prices = trader_data.get("resin_prices", [])
                self.resin_vwap = trader_data.get("resin_vwap", [])
                self.kelp_prices = trader_data.get("kelp_prices", [])
                self.kelp_vwap = trader_data.get("kelp_vwap", [])
            except:
                pass
        
        # Process Rainforest Resin orders
        if "RAINFOREST_RESIN" in state.order_depths:
            resin_position = state.position.get("RAINFOREST_RESIN", 0)
            position_limit = self.position_limits["RAINFOREST_RESIN"]
            
            # For Resin we use a fixed fair value since it's extremely stable
            resin_orders = self.resin_orders(
                state.order_depths["RAINFOREST_RESIN"], 
                resin_fair_value, 
                resin_width,
                resin_position, 
                position_limit
            )
            
            if resin_orders:
                result["RAINFOREST_RESIN"] = resin_orders
        
        # Process Kelp orders
        if "KELP" in state.order_depths:
            kelp_position = state.position.get("KELP", 0)
            position_limit = self.position_limits["KELP"]
            
            # For Kelp we use a dynamic fair value with VWAP
            kelp_orders = self.kelp_orders(
                state.order_depths["KELP"],
                kelp_timespan, 
                kelp_make_width, 
                kelp_take_width,
                kelp_position,
                position_limit
            )
            
            if kelp_orders:
                result["KELP"] = kelp_orders
        
        # Prepare trader data for next round
        trader_data = json.dumps({
            "resin_prices": self.resin_prices,
            "resin_vwap": self.resin_vwap,
            "kelp_prices": self.kelp_prices,
            "kelp_vwap": self.kelp_vwap,
            "day": self.day
        })
        
        # Log and return
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
