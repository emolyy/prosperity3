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
        orders = []
    
        # Check for valid order book presence
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders
        
        # Get market prices
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) // 2
        spread = best_ask - best_bid
        
        # Initialize price history if first run
        if not hasattr(self, 'squid_price_history'):
            self.squid_price_history = []
            self.squid_position_history = []
            self.squid_last_signal = 0
            self.squid_z_threshold = 2.2  # Z-score threshold for mean reversion signals (increased)
            self.squid_window_size = 40   # Window size for calculating mean and std dev (increased)
    
            # Add current price to history
        self.squid_price_history.append(mid_price)
        self.squid_position_history.append(position)
        
        # Need enough data to proceed
        if len(self.squid_price_history) < self.squid_window_size + 5:
            return orders
        
        # Calculate rolling window statistics
        window = self.squid_price_history[-self.squid_window_size:]
        mean_price = sum(window) / len(window)
        variance = sum((p - mean_price) ** 2 for p in window) / len(window)
        std_dev = max(1, (variance) ** 0.5)  # Floor of 1 to prevent extreme z-scores
        
        # Calculate z-score of current price
        z_score = (mid_price - mean_price) / std_dev
        
        # Position limits
        position_limit = self.position_limits.get("SQUID_INK", 50)
        buy_capacity = position_limit - position
        sell_capacity = position_limit + position
        
        # Generate mean reversion signal
        signal = 0
        signal_strength = 0
        
        # Check recent price direction - are we in a strong trend?
        price_trend = 0
        if len(self.squid_price_history) > 10:
            recent_prices = self.squid_price_history[-10:]
            price_trend = 1 if recent_prices[-1] > recent_prices[0] else -1
        
        # Only generate mean reversion signals when z-score is extreme
        # AND when the price is starting to move in the reversal direction
        if z_score < -self.squid_z_threshold:
            # Price significantly below mean - potential BUY
            # Check if we're seeing signs of upward movement
            if len(self.squid_price_history) > 3 and self.squid_price_history[-1] > self.squid_price_history[-2]:
                signal = 1
                signal_strength = min(1.0, abs(z_score) / 4)
        elif z_score > self.squid_z_threshold:
            # Price significantly above mean - potential SELL
            # Check if we're seeing signs of downward movement
            if len(self.squid_price_history) > 3 and self.squid_price_history[-1] < self.squid_price_history[-2]:
                signal = -1
                signal_strength = min(1.0, abs(z_score) / 4)
        
        # Execution logic - more conservative position sizing
        base_size = 3  # Even smaller base position size
            
        # Scale size based on how extreme the z-score is and confirmation signals
        if abs(z_score) > 4.0:
            # Very extreme deviation with confirmation - use larger size
            adjusted_size = 12
        elif abs(z_score) > 3.5:
            adjusted_size = 8
        elif abs(z_score) > 3.0:
            adjusted_size = 5
        else:
            adjusted_size = base_size
        
        # Only trade when there's enough liquidity to avoid poor execution
        # Skip trading if liquidity is too thin
        bid_liquidity = sum(order_depth.buy_orders.values())
        ask_liquidity = abs(sum(order_depth.sell_orders.values()))
        min_liquidity_threshold = 10
        
        if signal == 1 and ask_liquidity < min_liquidity_threshold:
            signal = 0  # Not enough sell orders to buy from
        elif signal == -1 and bid_liquidity < min_liquidity_threshold:
            signal = 0  # Not enough buy orders to sell to
        
        # Execute trades based on signal
        if signal == 1 and buy_capacity > 0:  # BUY
            # Position size based on available capacity and signal strength
            size = min(
                adjusted_size,
                buy_capacity,
                -order_depth.sell_orders[best_ask]  # Available volume
            )
            
            if size > 0:
                orders.append(Order("SQUID_INK", best_ask, size))
        
        elif signal == -1 and sell_capacity > 0:  # SELL
            # Position size based on available capacity and signal strength
            size = min(
                adjusted_size,
                sell_capacity,
                order_depth.buy_orders[best_bid]  # Available volume
            )
            
            if size > 0:
                orders.append(Order("SQUID_INK", best_bid, -size))
        
        # Opportunistic trading - take advantage of extreme price dislocations only
        # Use much stricter criteria
        if best_ask < mean_price - 3.5 * std_dev and buy_capacity > 0:
            # Only buy at extremely low prices
            size = min(
                4,  # Small position
                buy_capacity,
                -order_depth.sell_orders[best_ask]
            )
            if size > 0:
                orders.append(Order("SQUID_INK", best_ask, size))
        
        if best_bid > mean_price + 3.5 * std_dev and sell_capacity > 0:
            # Only sell at extremely high prices
            size = min(
                4,  # Small position
                sell_capacity,
                order_depth.buy_orders[best_bid]
            )
            if size > 0:
                orders.append(Order("SQUID_INK", best_bid, -size))
        
        # Risk management - more aggressive
        if abs(position) > position_limit * 0.6:  # Even lower threshold to prevent hitting limits
            # More aggressively reduce positions near limits
            if position > 0:
                # Long position - sell some regardless of price
                reduce_size = min(
                    max(3, int(position * 0.35)),  # Reduce by 35%, minimum 3 units
                    order_depth.buy_orders[best_bid]
                )
                if reduce_size > 0:
                    orders.append(Order("SQUID_INK", best_bid, -reduce_size))
            
            elif position < 0:
                # Short position - buy some regardless of price
                reduce_size = min(
                    max(3, int(-position * 0.35)),  # Reduce by 35%, minimum 3 units
                    -order_depth.sell_orders[best_ask]
                )
                if reduce_size > 0:
                    orders.append(Order("SQUID_INK", best_ask, reduce_size))
        
        # Enhanced aggressive profit-taking and stop-loss
        # Take profit earlier and more aggressively
        if position > 5 and best_bid > mean_price + 0.8 * std_dev:
            # Long position with price above mean - take profit
            take_profit_size = min(
                max(2, int(position * 0.6)),  # Take profit on 60% of position, minimum 2 units
                order_depth.buy_orders[best_bid]
            )
            if take_profit_size > 0:
                orders.append(Order("SQUID_INK", best_bid, -take_profit_size))
                
        elif position < -5 and best_ask < mean_price - 0.8 * std_dev:
            # Short position with price below mean - take profit
            take_profit_size = min(
                max(2, int(-position * 0.6)),  # Take profit on 60% of position, minimum 2 units
                -order_depth.sell_orders[best_ask]
            )
            if take_profit_size > 0:
                orders.append(Order("SQUID_INK", best_ask, take_profit_size))
        
        # Stop-loss logic for positions moving against us
        # If we're holding inventory and price is moving strongly against us, cut losses
        if position > 10 and best_bid < mean_price - 2 * std_dev:
            # Long position with price falling well below mean - cut losses
            stop_size = min(
                max(2, int(position * 0.7)),  # Cut up to 70% of position
                order_depth.buy_orders[best_bid]
            )
            if stop_size > 0:
                orders.append(Order("SQUID_INK", best_bid, -stop_size))
                
        elif position < -10 and best_ask > mean_price + 2 * std_dev:
            # Short position with price rising well above mean - cut losses
            stop_size = min(
                max(2, int(-position * 0.7)),  # Cut up to 70% of position
                -order_depth.sell_orders[best_ask]
            )
            if stop_size > 0:
                orders.append(Order("SQUID_INK", best_ask, stop_size))
        
        # Store last signal for next iteration
        self.squid_last_signal = signal
        
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