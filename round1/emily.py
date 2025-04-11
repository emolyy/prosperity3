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
        
        # Simple mean reversion parameters
        self.window_size = 30  # Rolling window for mean/std
        self.std_threshold = 1.5  # Number of std devs for signals

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
        orders = []
    
        # Check for valid order book presence
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return orders
        
        # Calculate current price metrics
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) // 2
        spread = best_ask - best_bid
        
        # Initialize data structures if first run
        if not hasattr(self, 'squid_data'):
            self.squid_data = {
                'prices': [],              # Price history
                'returns': [],             # Price returns
                'volatility': 0,           # Current volatility estimate
                'regime': 'unknown',       # Current market regime
                'regime_count': 0,         # How long we've been in current regime
                'last_signal': 0,          # Last trading signal
                'signal_strength': 0,      # Strength of current signal
                'turnover': 0,             # Trading turnover counter
                'success_trend': 0,        # Success counter for trend following
                'success_mean_rev': 0      # Success counter for mean reversion
            }
        
        # Add current price to history
        self.squid_data['prices'].append(mid_price)
        
        # Need enough data to proceed
        if len(self.squid_data['prices']) < 20:
            return orders
        
        # Calculate returns
        self.squid_data['returns'].append(
            (self.squid_data['prices'][-1] / self.squid_data['prices'][-2] - 1) 
            if len(self.squid_data['returns']) > 0 else 0
        )
        
        # Update volatility estimate using exponential moving average
        if len(self.squid_data['returns']) > 5:
            alpha = 0.2  # Weight for new data
            current_vol = abs(self.squid_data['returns'][-1]) * 100  # Convert to percentage
            self.squid_data['volatility'] = (
                alpha * current_vol + 
                (1 - alpha) * self.squid_data['volatility']
            )
        
        # ---- REGIME DETECTION ----
        # Calculate autocorrelation of returns to detect regime
        returns_window = self.squid_data['returns'][-20:]
        
        # Lag-1 autocorrelation (negative suggests mean reversion, positive suggests trend)
        if len(returns_window) > 5:
            mean_return = sum(returns_window) / len(returns_window)
            numerator = sum((returns_window[i] - mean_return) * (returns_window[i-1] - mean_return) 
                            for i in range(1, len(returns_window)))
            denominator = sum((r - mean_return) ** 2 for r in returns_window)
            
            if denominator > 0:
                autocorr = numerator / denominator
            else:
                autocorr = 0
                
            # Determine regime based on autocorrelation and recent directional movement
            price_direction = 1 if self.squid_data['prices'][-1] > self.squid_data['prices'][-5] else -1
            
            # Check if we need to change regime
            prev_regime = self.squid_data['regime']
            
            if autocorr < -0.2:
                new_regime = 'mean_reverting'
            elif autocorr > 0.2:
                new_regime = 'trending'
            else:
                # In the uncertain zone, give preference to the regime that's been working better
                if self.squid_data['success_trend'] > self.squid_data['success_mean_rev']:
                    new_regime = 'trending'
                else:
                    new_regime = 'mean_reverting'
            
            # Only change regime if we have a strong signal or have been wrong multiple times
            if prev_regime != new_regime:
                self.squid_data['regime'] = new_regime
                self.squid_data['regime_count'] = 0
            else:
                self.squid_data['regime_count'] += 1
        else:
            # Default to mean reversion if not enough data
            self.squid_data['regime'] = 'mean_reverting'
        
        # Position limits and available capacity
        position_limit = self.position_limits.get("SQUID_INK", 50)
        buy_capacity = position_limit - position
        sell_capacity = position_limit + position
        
        # ---- SIGNAL GENERATION ----
        signal = 0          # Trading signal: 1 for buy, -1 for sell, 0 for neutral
        signal_strength = 0 # Signal strength between 0 and 1
        
        # Different signals based on detected regime
        if self.squid_data['regime'] == 'mean_reverting':
            # Mean reversion logic
            # Calculate z-score of current price vs recent window
            window = self.squid_data['prices'][-15:]
            mean_price = sum(window) / len(window)
            
            # Calculate standard deviation with a floor to prevent division by zero
            variance = sum((p - mean_price) ** 2 for p in window) / len(window)
            std_dev = max(3, (variance) ** 0.5)  # Floor of 3 to prevent extreme z-scores
            
            z_score = (mid_price - mean_price) / std_dev
            
            # Generate mean reversion signal
            if z_score < -1.0:
                # Price significantly below mean - BUY signal
                signal = 1
                signal_strength = min(1.0, abs(z_score) / 3)  # Cap at 1.0
            elif z_score > 1.0:
                # Price significantly above mean - SELL signal
                signal = -1
                signal_strength = min(1.0, abs(z_score) / 3)  # Cap at 1.0
        
        else:  # Trending regime
            # Moving average crossover for trend following
            short_ma = sum(self.squid_data['prices'][-5:]) / 5
            long_ma = sum(self.squid_data['prices'][-15:]) / 15
            
            # Normalized price momentum (recent return)
            momentum = ((self.squid_data['prices'][-1] / self.squid_data['prices'][-5]) - 1) * 100
            
            # Generate trend following signal
            if short_ma > long_ma * 1.002:
                # Uptrend signal
                signal = 1
                signal_strength = min(1.0, momentum / 2)  # Cap at 1.0
            elif short_ma < long_ma * 0.998:
                # Downtrend signal
                signal = -1
                signal_strength = min(1.0, abs(momentum) / 2)  # Cap at 1.0
        
        # Store current signal
        self.squid_data['last_signal'] = signal
        self.squid_data['signal_strength'] = signal_strength
        
        # ---- POSITION SIZING ----
        # Size based on signal strength and available capacity
        # Stronger signals use more capacity
        
        # Base position size (10-40% of capacity based on signal strength)
        base_size_pct = 0.1 + (0.3 * signal_strength)
        
        # Adjust further based on volatility (reduce size in high volatility)
        adjusted_size_pct = base_size_pct / (1 + (self.squid_data['volatility'] / 10))
        
        # Scale position based on how correct our recent signals have been
        if self.squid_data['regime'] == 'trending' and self.squid_data['success_trend'] > 0:
            adjusted_size_pct *= min(2.0, 1.0 + (self.squid_data['success_trend'] / 10))
        elif self.squid_data['regime'] == 'mean_reverting' and self.squid_data['success_mean_rev'] > 0:
            adjusted_size_pct *= min(2.0, 1.0 + (self.squid_data['success_mean_rev'] / 10))
        
        # ---- ORDER EXECUTION ----
        # Only trade if signal is non-zero and we have capacity
        if signal == 1 and buy_capacity > 0:  # BUY
            # Position size based on available capacity and size percentage
            size = min(
                int(buy_capacity * adjusted_size_pct),
                -order_depth.sell_orders[best_ask]  # Available volume
            )
            
            if size > 0:
                orders.append(Order("SQUID_INK", best_ask, size))
                self.squid_data['turnover'] += size
        
        elif signal == -1 and sell_capacity > 0:  # SELL
            # Position size based on available capacity and size percentage
            size = min(
                int(sell_capacity * adjusted_size_pct),
                order_depth.buy_orders[best_bid]  # Available volume
            )
            
            if size > 0:
                orders.append(Order("SQUID_INK", best_bid, -size))
                self.squid_data['turnover'] += size
        
        # ---- RISK MANAGEMENT ----
        # Reduce large positions that go against current signal
        if position > position_limit * 0.7 and signal < 0:
            # We are heavily long but signal is negative - reduce position
            reduce_size = min(
                int(position * 0.2),  # Reduce by 20%
                order_depth.buy_orders[best_bid]
            )
            if reduce_size > 0:
                orders.append(Order("SQUID_INK", best_bid, -reduce_size))
                self.squid_data['turnover'] += reduce_size
        
        elif position < -position_limit * 0.7 and signal > 0:
            # We are heavily short but signal is positive - reduce position
            reduce_size = min(
                int(-position * 0.2),  # Reduce by 20%
                -order_depth.sell_orders[best_ask]
            )
            if reduce_size > 0:
                orders.append(Order("SQUID_INK", best_ask, reduce_size))
                self.squid_data['turnover'] += reduce_size
        
        # ---- STRATEGY REINFORCEMENT LEARNING ----
        # Update success counters based on P&L (simple learning mechanism)
        if len(self.squid_data['prices']) > 5:
            price_change = self.squid_data['prices'][-1] - self.squid_data['prices'][-2]
            prev_signal = self.squid_data['last_signal']
            
            # If our signal matches the price change direction, count a success
            if (prev_signal > 0 and price_change > 0) or (prev_signal < 0 and price_change < 0):
                if self.squid_data['regime'] == 'trending':
                    self.squid_data['success_trend'] = min(10, self.squid_data['success_trend'] + 1)
                else:
                    self.squid_data['success_mean_rev'] = min(10, self.squid_data['success_mean_rev'] + 1)
            # If our signal was wrong, reduce success counter
            elif (prev_signal > 0 and price_change < 0) or (prev_signal < 0 and price_change > 0):
                if self.squid_data['regime'] == 'trending':
                    self.squid_data['success_trend'] = max(0, self.squid_data['success_trend'] - 1)
                else:
                    self.squid_data['success_mean_rev'] = max(0, self.squid_data['success_mean_rev'] - 1)
        
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