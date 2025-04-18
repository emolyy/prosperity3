from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any, Dict
import jsonpickle
import numpy as np
import json
import numpy as np

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
            [], #self.compress_trades(state.market_trades),
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

    def truncate(self, value, max_length):
        # Handle negative max_length values properly
        if max_length < 0:
            # Ensure we don't try to slice beyond the string length
            effective_length = max(0, len(str(value)) + max_length - 3)
            return str(value)[:effective_length] + "..." if len(str(value)) > abs(max_length) else value
        
        # Original logic for positive max_length
        if len(str(value)) <= max_length:
            return value
        return str(value)[: max_length - 3] + "..."

logger = Logger()

class Product:
    SQUID_INK = "SQUID_INK"

    PARAMS = {
        SQUID_INK: {
            "take_width": 3.0,  # Increased from 2.5 to capture more profitable trades
            "clear_width": 0.5,  # Reduced from 0.75 for more aggressive position clearing
            "prevent_adverse": True,  # Changed from False to avoid adverse selection
            "adverse_volume": 20,  # Increased from 15 to be more selective
            "reversion_beta": -0.4,  # Strengthened from -0.35 to capitalize on reversals
            "disregard_edge": 2,  # Increased from 1 to be more selective with orders
            "join_edge": 1,  # Increased from 0 to join favorable prices
            "default_edge": 1.5,  # Increased from 1 for better profitability
            "cycle_lookback": 30,  # Increased from 20 for better pattern detection
            "volatility_adjustment": 0.7,  # Increased from 0.5 for more dynamic width
            "soft_position_limit": 20,  # Reduced from 25 for tighter position management
            "pattern_detection": True,  # New parameter to enable pattern detection
            "momentum_factor": 0.3,  # New parameter for momentum consideration
        },
    }

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = Product.PARAMS
        self.params = params
        self.LIMIT = {
            Product.SQUID_INK: 50,
        }
        self.price_history = {
            Product.SQUID_INK: [],
        }
        
    def detect_cycle_position(self, product, price_history, current_price):
        """Enhanced cycle detection with pattern recognition"""
        if len(price_history) < self.params[product]["cycle_lookback"]:
            return 0  # Neutral if not enough history
        
        # Calculate recent high and low
        recent_prices = price_history[-self.params[product]["cycle_lookback"]:]
        recent_high = max(recent_prices)
        recent_low = min(recent_prices)
        price_range = recent_high - recent_low
        
        if price_range == 0:
            return 0  # Avoid division by zero
        
        # Normalize current price position in the range [-1, 1]
        position_in_cycle = 2 * (current_price - recent_low) / price_range - 1
        
        # Pattern recognition - detect trend direction
        if len(recent_prices) >= 10:
            short_term_trend = sum(recent_prices[-5:]) / 5 - sum(recent_prices[-10:-5]) / 5
            # Adjust cycle position based on trend direction
            if short_term_trend > 0:
                position_in_cycle = min(1, position_in_cycle + 0.2)  # Boost upward trend signal
            elif short_term_trend < 0:
                position_in_cycle = max(-1, position_in_cycle - 0.2)  # Boost downward trend signal
        
        return position_in_cycle


    def calculate_dynamic_reversion(self, product, cycle_position, price_history):
        """Calculate a more sophisticated dynamic reversion factor"""
        base_reversion = self.params[product]["reversion_beta"]
        
        # Stronger mean reversion when price is at extremes
        cycle_adjustment = abs(cycle_position) * 0.3
        
        # Detect acceleration/deceleration in price movements
        if len(price_history) >= 15:
            recent_returns = [
                (price_history[i] - price_history[i-1]) / price_history[i-1] 
                for i in range(-14, 0)
            ]
            
            # Calculate momentum
            momentum = sum(recent_returns[-5:]) / sum(recent_returns[-10:-5]) if sum(recent_returns[-10:-5]) != 0 else 0
            
            # Adjust reversion based on momentum
            momentum_factor = 0.2 * (1 - min(1, abs(momentum)))
            
            # Return adjusted reversion factor with momentum consideration
            return base_reversion * (1 + cycle_adjustment + momentum_factor)
        
        # Default if not enough history
        return base_reversion * (1 + cycle_adjustment)
    
    def calculate_fair_value(self, product: str, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            
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
            
            if mm_ask is None or mm_bid is None:
                if traderObject.get(f"{product}_last_price", None) is None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject[f"{product}_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            
            # Store price in history for cycle detection
            if product in self.price_history:
                self.price_history[product].append(mmmid_price)
                # Keep history at a reasonable size
                if len(self.price_history[product]) > 100:
                    self.price_history[product] = self.price_history[product][-100:]
            
            # Detect cycle position
            cycle_position = self.detect_cycle_position(
                product, 
                self.price_history[product], 
                mmmid_price
            )
            
            # Calculate dynamic reversion factor
            dynamic_reversion = self.calculate_dynamic_reversion(
                product, 
                cycle_position,
                self.price_history[product]  # Add the missing price_history parameter
            )
            
            if traderObject.get(f"{product}_last_price", None) is not None:
                last_price = traderObject[f"{product}_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                
                # Use dynamic reversion factor instead of fixed one
                pred_returns = last_returns * dynamic_reversion
                
                # Store cycle information for position management
                traderObject[f"{product}_cycle_position"] = cycle_position
                
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
                
            traderObject[f"{product}_last_price"] = mmmid_price
            return fair
        return None

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> tuple[List[Order], int, int]:
        position_limit = self.LIMIT[product]
        
        # Adjust take_width based on cycle position if available
        cycle_position = 0
        if f"{product}_cycle_position" in traderObject:
            cycle_position = traderObject[f"{product}_cycle_position"]
            
        # Asymmetric take width based on cycle position
        buy_take_width = take_width * (1 + 0.5 * max(0, -cycle_position))  # Wider when price is low
        sell_take_width = take_width * (1 + 0.5 * max(0, cycle_position))  # Wider when price is high

        # Use a loop instead of recursion
        while True:
            orders_processed = False
            
            # Process sell orders (buying opportunities)
            sell_prices = sorted(order_depth.sell_orders.keys())  # Sort by price (ascending)
            for ask_price in sell_prices:
                if ask_price > fair_value - buy_take_width:
                    break  # No more favorable prices
                    
                ask_amount = -1 * order_depth.sell_orders[ask_price]
                if prevent_adverse and abs(ask_amount) > adverse_volume:
                    continue  # Skip this order due to adverse selection
                    
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
            buy_prices = sorted(order_depth.buy_orders.keys(), reverse=True)  # Sort by price (descending)
            for bid_price in buy_prices:
                if bid_price < fair_value + sell_take_width:
                    break  # No more favorable prices
                    
                bid_amount = order_depth.buy_orders[bid_price]
                if prevent_adverse and abs(bid_amount) > adverse_volume:
                    continue  # Skip this order due to adverse selection
                    
                quantity = min(bid_amount, position_limit + position - sell_order_volume)
                if quantity <= 0:
                    break  # Position limit reached
                    
                orders.append(Order(product, bid_price, -1 * quantity))
                sell_order_volume += quantity
                order_depth.buy_orders[bid_price] -= quantity
                orders_processed = True
                
                if order_depth.buy_orders[bid_price] == 0:
                    del order_depth.buy_orders[bid_price]
            
            # If no orders were processed in this iteration, we're done
            if not orders_processed:
                break
                
        return orders, buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        price_history=None,
    ):
        # Get cycle position
        cycle_position = traderObject.get(f"{product}_cycle_position", 0)
        
        # Get pattern signals
        buy_signal, sell_signal = self.identify_pattern_signals(
            product, 
            price_history or self.price_history.get(product, []),
            cycle_position
        )
        
        # Base position sizing
        base_quantity = int(self.LIMIT[product] * 0.5)
        
        # Adjust quantities based on cycle position and signals
        buy_adjustment = max(0, -cycle_position) * 10  # Buy more when price is low
        sell_adjustment = max(0, cycle_position) * 10  # Sell more when price is high
        
        # Apply signal boosts
        if buy_signal > 0:
            buy_adjustment += buy_signal * 5  # Boost buy size on buy signals
        if sell_signal > 0:
            sell_adjustment += sell_signal * 5  # Boost sell size on sell signals
        
        # Apply volatility-based sizing if we have price history
        if price_history or self.price_history.get(product, []):
            base_quantity = self.adjust_position_size(
                product,
                base_quantity,
                price_history or self.price_history.get(product, []),
                cycle_position
            )
        
        # Calculate final quantities
        buy_quantity = min(
            self.LIMIT[product] - (position + buy_order_volume),
            base_quantity + int(buy_adjustment)
        )
        
        sell_quantity = min(
            self.LIMIT[product] + (position - sell_order_volume),
            base_quantity + int(sell_adjustment)
        )
        
        # Adjust bid/ask based on signals
        if buy_signal > 1:
            bid += 1  # More aggressive bid on strong buy signals
        if sell_signal > 1:
            ask -= 1  # More aggressive ask on strong sell signals
        
        # Place orders
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))
        
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
        
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[int, int]:
        position_after_take = position + buy_order_volume - sell_order_volume
        
        # Adjust clearing width based on position size
        # More aggressive clearing for larger positions
        position_factor = min(1, abs(position_after_take) / (self.LIMIT[product] * 0.7))
        adjusted_width = width * (1 - 0.5 * position_factor)  # Reduce width for larger positions
        
        fair_for_bid = round(fair_value - adjusted_width)
        fair_for_ask = round(fair_value + adjusted_width)
        
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        
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

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> tuple[List[Order], int, int]:
        orders: List[Order] = []
        
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = True,  # Changed to True
        soft_position_limit: int = 0,
    ):
        orders: List[Order] = []
        
        # Adjust edge based on cycle position if available
        cycle_position = 0
        if f"{product}_cycle_position" in traderObject:
            cycle_position = traderObject[f"{product}_cycle_position"]
            
        # Asymmetric edge based on cycle position
        buy_edge = default_edge * (1 - 0.3 * max(0, -cycle_position))  # Tighter when price is low
        sell_edge = default_edge * (1 - 0.3 * max(0, cycle_position))  # Tighter when price is high
        
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]
        
        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None
        
        ask = round(fair_value + sell_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1
                
        bid = round(fair_value - buy_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1
                
        # Enhanced position management
        if manage_position:
            # Use soft_position_limit from parameters
            limit = self.params[product]["soft_position_limit"]
            
            # Adjust quotes based on position and cycle position
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
                
        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        
        return orders, buy_order_volume, sell_order_volume
    
    def calculate_volatility(self, price_history, lookback=20):
        """Calculate recent price volatility"""
        if len(price_history) < lookback:
            return 0.01  # Default low volatility if not enough history
        
        recent_prices = price_history[-lookback:]
        returns = [(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] for i in range(1, len(recent_prices))]
        return np.std(returns) * np.sqrt(252)  # Annualized volatility

    def adjust_position_size(self, product, base_size, price_history, cycle_position):
        """Adjust position size based on volatility and cycle position"""
        volatility = self.calculate_volatility(price_history)
        
        # Base volatility adjustment - reduce size in high volatility
        vol_adjustment = max(0.5, 1 - volatility * 10)
        
        # Cycle position adjustment - increase size when favorable
        cycle_adjustment = 1 + 0.5 * (
            max(0, -cycle_position) if cycle_position < 0 else  # Increase buys at cycle lows
            max(0, cycle_position) if cycle_position > 0 else   # Increase sells at cycle highs
            0
        )
        
        return int(base_size * vol_adjustment * cycle_adjustment)   

    def identify_pattern_signals(self, product, price_history, cycle_position):
        """Identify specific entry/exit signals based on price patterns"""
        if len(price_history) < 30:
            return 0, 0  # No signals if insufficient history
        
        # Calculate moving averages
        ma_short = sum(price_history[-10:]) / 10
        ma_medium = sum(price_history[-20:]) / 20
        ma_long = sum(price_history[-30:]) / 30
        
        # Calculate price momentum
        momentum = (price_history[-1] - price_history[-5]) / price_history[-5]
        
        # Buy signal conditions
        buy_signal = 0
        if ma_short > ma_medium and ma_medium > ma_long and cycle_position < -0.3:
            # Upward trend confirmation at cycle low
            buy_signal = 1
        elif momentum < -0.02 and cycle_position < -0.7:
            # Strong oversold condition
            buy_signal = 2
        
        # Sell signal conditions
        sell_signal = 0
        if ma_short < ma_medium and ma_medium < ma_long and cycle_position > 0.3:
            # Downward trend confirmation at cycle high
            sell_signal = 1
        elif momentum > 0.02 and cycle_position > 0.7:
            # Strong overbought condition
            sell_signal = 2
        
        return buy_signal, sell_signal
 

    def run(self, state: TradingState):
        global traderObject
        traderObject = {}
        
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
            
        result = {}
        conversions = 0
        trader_data = ""
        
        # Initialize orders as an empty list before using it
        orders = []
        
        # Add Round 1 products trading
        for product in [Product.SQUID_INK]:
            if product in state.order_depths:
                position = state.position.get(product, 0)
                
                # Initialize buy_order_volume and sell_order_volume
                buy_order_volume = 0
                sell_order_volume = 0
                
                # Calculate fair value based on product type
                fair_value = self.calculate_fair_value(product, state.order_depths[product], traderObject)
                
                if fair_value is not None:
                    # Take orders
                    take_orders, buy_order_volume, sell_order_volume = self.take_best_orders(
                        product,
                        fair_value,
                        self.params[product]["take_width"],
                        orders,
                        state.order_depths[product],
                        position,
                        buy_order_volume,
                        sell_order_volume,
                        self.params[product].get("prevent_adverse", False),
                        self.params[product].get("adverse_volume", 0),
                    )
                    
                    # Clear position
                    clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                        product,
                        state.order_depths[product],
                        fair_value,
                        self.params[product]["clear_width"],
                        position,
                        buy_order_volume,
                        sell_order_volume,
                    )
                    
                    # Make orders
                    make_orders, _, _ = self.make_orders(
                        product,
                        state.order_depths[product],
                        fair_value,
                        position,
                        buy_order_volume,
                        sell_order_volume,
                        self.params[product]["disregard_edge"],
                        self.params[product]["join_edge"],
                        self.params[product]["default_edge"],
                        True,  # Always manage position
                        self.params[product]["soft_position_limit"],
                    )
                    
                    result[product] = take_orders + clear_orders + make_orders
        
        trader_data = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data