from typing import Dict, List, Any
import json
import numpy as np
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
        self.position_limits = {"SQUID_INK": 50}
        
        # Price history storage
        self.price_history = {"SQUID_INK": []}
        
        # Strategy parameters - easily adjustable
        self.params = {
            "SQUID_INK": {
                # Early trading parameters
                "early_width": 1.5,  # Reduced from 2.0
                "early_position_size": 4,  # Reduced from 5
                "bootstrap_threshold": 20,  # Reduced from 30
                
                # Advanced strategy parameters
                "take_width": 2.0,  # Reduced from 3.0
                "clear_width": 0.7,  # Increased from 0.5
                "prevent_adverse": True,
                "adverse_volume": 15,  # Reduced from 20
                "reversion_beta": -0.35,  # Reduced from -0.45
                "disregard_edge": 1.5,  # Reduced from 2
                "join_edge": 0.8,  # Reduced from 1
                "default_edge": 1.3,  # Reduced from 1.5
                "cycle_lookback": 75,  # Increased from 40
                "volatility_adjustment": 0.9,  # Increased from 0.8
                "soft_position_limit": 50,  # Reduced from 35
                
                # New parameters
                "trend_factor": 0.3,
                "momentum_lookback": 10,
                "max_position_at_extreme": 15
            }
        }
        
        # Strategy state tracking
        self.strategy_state = {}
        
    def calculate_midpoint(self, order_depth: OrderDepth) -> float:
        """Calculate the midpoint price from the order book."""
        if len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_ask + best_bid) / 2
        return None
    
    def bootstrap_strategy(self, product: str, order_depth: OrderDepth, position: int) -> List[Order]:
        """Simple market making strategy for early iterations."""
        orders = []
        
        mid_price = self.calculate_midpoint(order_depth)
        if mid_price is None:
            return orders
        
        # Store price
        self.price_history[product].append(mid_price)
        
        # Calculate dynamic width based on recent volatility
        width = self.params[product]["early_width"]
        if len(self.price_history[product]) >= 5:
            recent_prices = self.price_history[product][-5:]
            std_dev = np.std(recent_prices)
            width = max(1.0, min(3.0, std_dev * 2))
        
        # Calculate position size (start small, increase gradually)
        progress_factor = min(1.0, len(self.price_history[product]) / self.params[product]["bootstrap_threshold"])
        position_size = int(self.params[product]["early_position_size"] + 
                          (15 * progress_factor))  # Scale from 5 to 20
        
        # Simple market making
        bid_price = int(mid_price - width)
        ask_price = int(mid_price + width)
        
        # Adjust for current position
        if position > 15:
            ask_price -= 1  # More aggressive selling when long
        elif position < -15:
            bid_price += 1  # More aggressive buying when short
        
        # Basic opportunistic trading
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        
        # Opportunistic buying
        if best_ask < mid_price - width * 1.5:
            buy_qty = min(-order_depth.sell_orders[best_ask], 
                         self.position_limits[product] - position)
            if buy_qty > 0:
                orders.append(Order(product, best_ask, buy_qty))
        
        # Opportunistic selling
        if best_bid > mid_price + width * 1.5:
            sell_qty = min(order_depth.buy_orders[best_bid], 
                          self.position_limits[product] + position)
            if sell_qty > 0:
                orders.append(Order(product, best_bid, -sell_qty))
        
        # Regular market making
        buy_qty = min(position_size, self.position_limits[product] - position)
        if buy_qty > 0:
            orders.append(Order(product, bid_price, buy_qty))
        
        sell_qty = min(position_size, self.position_limits[product] + position)
        if sell_qty > 0:
            orders.append(Order(product, ask_price, -sell_qty))
        
        return orders
    
    def detect_cycle_position(self, product, price_history, current_price):
        """Enhanced cycle detection with multiple timeframes."""
        lookback = self.params[product]["cycle_lookback"]
        if len(price_history) < lookback:
            return 0
        
        # Primary cycle detection (longer timeframe)
        recent_prices = price_history[-lookback:]
        recent_high = max(recent_prices)
        recent_low = min(recent_prices)
        price_range = recent_high - recent_low
        
        if price_range == 0:
            return 0
        
        # Calculate position in cycle (-1 to 1 range)
        position_in_cycle = 2 * (current_price - recent_low) / price_range - 1
        
        # Add shorter timeframe confirmation
        if len(recent_prices) >= 25:
            short_lookback = 25
            short_prices = price_history[-short_lookback:]
            short_high = max(short_prices)
            short_low = min(short_prices)
            short_range = short_high - short_low
            
            if short_range > 0:
                short_position = 2 * (current_price - short_low) / short_range - 1
                # Blend long and short timeframe signals
                position_in_cycle = 0.7 * position_in_cycle + 0.3 * short_position
        
        return max(-1, min(1, position_in_cycle))
    
    def calculate_atr(self, price_history, lookback=14):
        """Calculate Average True Range for volatility assessment."""
        if len(price_history) < lookback:
            return 0.01
        
        tr_values = []
        for i in range(1, len(price_history)):
            high = price_history[i]
            low = price_history[i-1]
            tr = abs(high - low)
            tr_values.append(tr)
        
        atr = np.mean(tr_values[-lookback:])
        return atr
    
    def calculate_fair_value(self, product: str, order_depth: OrderDepth) -> float:
        """Calculate fair value based on order book and historical data."""
        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return None
        
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        
        # Filter large orders that might be market makers
        filtered_ask = [
            price
            for price in order_depth.sell_orders.keys()
            if abs(order_depth.sell_orders[price]) >= self.params[product]["adverse_volume"]
        ]
        
        filtered_bid = [
            price
            for price in order_depth.buy_orders.keys()
            if abs(order_depth.buy_orders[price]) >= self.params[product]["adverse_volume"]
        ]
        
        mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
        mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
        
        # Calculate midpoint price considering market maker orders
        if mm_ask is None or mm_bid is None:
            if product + "_last_price" not in self.strategy_state:
                mmmid_price = (best_ask + best_bid) / 2
            else:
                mmmid_price = self.strategy_state[product + "_last_price"]
        else:
            mmmid_price = (mm_ask + mm_bid) / 2
        
        # Store price in history
        self.price_history[product].append(mmmid_price)
        if len(self.price_history[product]) > 100:
            self.price_history[product] = self.price_history[product][-100:]
        
        # Calculate cycle position
        cycle_position = self.detect_cycle_position(
            product, 
            self.price_history[product], 
            mmmid_price
        )
        
        # Store cycle position for later use
        self.strategy_state[product + "_cycle_position"] = cycle_position
        
        # Apply mean reversion logic if we have previous price
        if product + "_last_price" in self.strategy_state:
            last_price = self.strategy_state[product + "_last_price"]
            
            # Calculate dynamic reversion factor
            base_reversion = self.params[product]["reversion_beta"]
            cycle_adjustment = abs(cycle_position) * 0.3
            dynamic_reversion = base_reversion * (1 + cycle_adjustment)
            
            # Calculate price change and prediction
            last_returns = (mmmid_price - last_price) / last_price
            pred_returns = last_returns * dynamic_reversion
            
            # Calculate fair value with mean reversion
            fair = mmmid_price + (mmmid_price * pred_returns)
        else:
            fair = mmmid_price
        
        # Store last price for next iteration
        self.strategy_state[product + "_last_price"] = mmmid_price
    
        # Add trend component
        momentum_lookback = self.params[product]["momentum_lookback"]
        if len(self.price_history[product]) >= momentum_lookback + 5:
            recent_avg = sum(self.price_history[product][-5:]) / 5
            past_avg = sum(self.price_history[product][-momentum_lookback-5:-5]) / momentum_lookback
            momentum = (recent_avg - past_avg) / past_avg
            
            # Scale momentum effect based on cycle position
            cycle_position = self.strategy_state.get(product + "_cycle_position", 0)
            cycle_factor = 1.0 - 0.5 * abs(cycle_position)
            
            # Adjust fair value with momentum component
            trend_adjustment = momentum * self.params[product]["trend_factor"] * cycle_factor
            fair = fair * (1 + trend_adjustment)
        
        return fair
    
    def take_best_orders(self, product: str, fair_value: float, take_width: float, 
                         orders: List[Order], order_depth: OrderDepth, position: int,
                         buy_order_volume: int, sell_order_volume: int,
                         prevent_adverse: bool = False, adverse_volume: int = 0) -> tuple:
        """Take advantageous orders from the order book."""
        position_limit = self.position_limits[product]
        
        # Process orders in a loop until no more opportunities
        while True:
            orders_processed = False
            
            # Process sell orders (buying opportunities)
            sell_prices = sorted(order_depth.sell_orders.keys())
            for ask_price in sell_prices:
                if ask_price > fair_value - take_width:
                    break  # No more favorable prices
                
                ask_amount = -1 * order_depth.sell_orders[ask_price]
                if prevent_adverse and abs(ask_amount) > adverse_volume:
                    continue  # Skip due to adverse selection risk
                
                quantity = min(ask_amount, position_limit - position - buy_order_volume)
                if quantity <= 0:
                    break  # Position limit reached
                
                orders.append(Order(product, ask_price, quantity))
                buy_order_volume += quantity
                order_depth.sell_orders[ask_price] += quantity
                orders_processed = True
                
                if order_depth.sell_orders[ask_price] == 0:
                    del order_depth.sell_orders[ask_price]
            
            # Process buy orders (selling opportunities)
            buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)
            for bid_price in buy_prices:
                if bid_price < fair_value + take_width:
                    break  # No more favorable prices
                
                bid_amount = order_depth.buy_orders[bid_price]
                if prevent_adverse and abs(bid_amount) > adverse_volume:
                    continue  # Skip due to adverse selection risk
                
                quantity = min(bid_amount, position_limit + position - sell_order_volume)
                if quantity <= 0:
                    break  # Position limit reached
                
                orders.append(Order(product, bid_price, -1 * quantity))
                sell_order_volume += quantity
                order_depth.buy_orders[bid_price] -= quantity
                orders_processed = True
                
                if order_depth.buy_orders[bid_price] == 0:
                    del order_depth.buy_orders[bid_price]
            
            # Exit loop if no orders processed
            if not orders_processed:
                break
        
        return orders, buy_order_volume, sell_order_volume
    
    def clear_position_order(self, product: str, fair_value: float, width: float,
                            orders: List[Order], order_depth: OrderDepth, position: int,
                            buy_order_volume: int, sell_order_volume: int) -> tuple:
        """Clear positions that are against our current fair value estimate."""
        position_after_take = position + buy_order_volume - sell_order_volume
        
        # Adjust clearing width based on position size
        position_factor = min(1, abs(position_after_take) / (self.position_limits[product] * 0.7))
        adjusted_width = width * (1 - 0.5 * position_factor)
        
        fair_for_bid = round(fair_value - adjusted_width)
        fair_for_ask = round(fair_value + adjusted_width)
        
        buy_quantity = self.position_limits[product] - (position + buy_order_volume)
        sell_quantity = self.position_limits[product] + (position - sell_order_volume)
        
        # If we're long, try to sell at a good price
        if position_after_take > 0:
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        
        # If we're short, try to buy at a good price
        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        
        return buy_order_volume, sell_order_volume
    
    def clear_orders(self, product: str, order_depth: OrderDepth, fair_value: float,
                    clear_width: float, position: int, buy_order_volume: int, 
                    sell_order_volume: int) -> tuple:
        """Wrapper for position clearing logic."""
        orders = []
        
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product, fair_value, clear_width, orders, order_depth,
            position, buy_order_volume, sell_order_volume
        )
        
        return orders, buy_order_volume, sell_order_volume
    
    def make_orders(self, product: str, order_depth: OrderDepth, fair_value: float,
                   position: int, buy_order_volume: int, sell_order_volume: int,
                   disregard_edge: float, join_edge: float, default_edge: float) -> tuple:
        """Create market making orders with sophisticated edge adjustment."""
        orders = []
        
        # Get cycle position if available
        cycle_position = 0
        if product + "_cycle_position" in self.strategy_state:
            cycle_position = self.strategy_state[product + "_cycle_position"]
        
        # Asymmetric edge based on cycle position
        buy_edge = default_edge * (1 - 0.3 * max(0, -cycle_position))
        sell_edge = default_edge * (1 - 0.3 * max(0, cycle_position))
        
        # Find prices to post at
        asks_above_fair = [
            price for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        
        bids_below_fair = [
            price for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]
        
        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None
        
        # Set our ask price, joining existing levels if good
        ask = round(fair_value + sell_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # Join at this level
            else:
                ask = best_ask_above_fair - 1  # Just undercut
        
        # Set our bid price, joining existing levels if good        
        bid = round(fair_value - buy_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair  # Join at this level
            else:
                bid = best_bid_below_fair + 1  # Just outbid
        
        # Position management - adjust quotes based on position
        limit = self.params[product]["soft_position_limit"]
        
        if position > limit:
            # More aggressive selling when long
            ask_adjustment = 1 + int(abs(position - limit) / 5)
            ask -= ask_adjustment
        elif position < -limit:
            # More aggressive buying when short
            bid_adjustment = 1 + int(abs(position + limit) / 5)
            bid += bid_adjustment
        
        # Additional adjustment based on cycle position
        if cycle_position > 0.5 and position > 0:
            # More aggressive selling at cycle peaks when long
            ask -= 1
        elif cycle_position < -0.5 and position < 0:
            # More aggressive buying at cycle troughs when short
            bid += 1
        
        # Calculate order quantities
        buy_quantity = self.position_limits[product] - (position + buy_order_volume)
        sell_quantity = self.position_limits[product] + (position - sell_order_volume)
        
        # Adjust quantities based on volatility and cycle position
        atr = self.calculate_atr(self.price_history[product])
        vol_adjustment = max(0.5, 1 - atr * 10)
        
        # Volume is higher when expecting a reversal
        cycle_factor = 1.0
        if cycle_position > 0.7 and position < 0:  # At peak and short
            cycle_factor = 0.5  # Reduce short position
        elif cycle_position < -0.7 and position > 0:  # At trough and long
            cycle_factor = 0.5  # Reduce long position
        elif cycle_position < -0.5 and position < 0:  # Near trough and short
            cycle_factor = 1.5  # Increase short position
        elif cycle_position > 0.5 and position > 0:  # Near peak and long
            cycle_factor = 1.5  # Increase long position
        
        buy_quantity = int(min(buy_quantity, max(1, buy_quantity * vol_adjustment * cycle_factor)))
        sell_quantity = int(min(sell_quantity, max(1, sell_quantity * vol_adjustment * cycle_factor)))
        
        # Place orders if quantities are positive
        if buy_quantity > 0:
            orders.append(Order(product, bid, buy_quantity))
        
        if sell_quantity > 0:
            orders.append(Order(product, ask, -sell_quantity))
        
        return orders, buy_order_volume, sell_order_volume
    
    def advanced_strategy(self, product: str, order_depth: OrderDepth, position: int) -> List[Order]:
        """Full advanced trading strategy based on squid_ink.py."""
        orders = []
        buy_order_volume = 0
        sell_order_volume = 0
        
        # Calculate fair value
        fair_value = self.calculate_fair_value(product, order_depth)
        if fair_value is None:
            return orders
        
        # 1. Take advantageous orders
        take_orders, buy_order_volume, sell_order_volume = self.take_best_orders(
            product, fair_value, self.params[product]["take_width"],
            [], order_depth, position, buy_order_volume, sell_order_volume,
            self.params[product]["prevent_adverse"], self.params[product]["adverse_volume"]
        )
        orders.extend(take_orders)
        
        # 2. Clear position when needed
        clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
            product, order_depth, fair_value, self.params[product]["clear_width"],
            position, buy_order_volume, sell_order_volume
        )
        orders.extend(clear_orders)
        
        # 3. Make new orders
        make_orders, _, _ = self.make_orders(
            product, order_depth, fair_value, position,
            buy_order_volume, sell_order_volume,
            self.params[product]["disregard_edge"],
            self.params[product]["join_edge"],
            self.params[product]["default_edge"]
        )
        orders.extend(make_orders)
        
        return orders
    
    def run(self, state: TradingState):
        # Load previous state if available
        if state.traderData and state.traderData != "":
            try:
                saved_data = json.loads(state.traderData)
                if "price_history" in saved_data:
                    self.price_history = saved_data["price_history"]
                if "strategy_state" in saved_data:
                    self.strategy_state = saved_data["strategy_state"]
            except:
                pass  # If loading fails, continue with current state
        
        result = {}
        
        for product in ["SQUID_INK"]:
            if product not in state.order_depths:
                continue
            
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            
            # Decide which strategy to use based on data availability
            price_history_length = len(self.price_history.get(product, []))
            bootstrap_threshold = self.params[product]["bootstrap_threshold"]
            
            if price_history_length < bootstrap_threshold:
                # Use simpler strategy for early iterations
                orders = self.bootstrap_strategy(product, order_depth, position)
            else:
                # Use advanced strategy once we have enough data
                orders = self.advanced_strategy(product, order_depth, position)
            
            if orders:
                result[product] = orders
        
        # Save state for next iteration
        trader_data = json.dumps({
            "price_history": self.price_history,
            "strategy_state": self.strategy_state
        })

        logger.flush(state, result, 0, trader_data)
        return result, 0, trader_data