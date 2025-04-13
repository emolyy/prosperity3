import json
import statistics
from typing import Any, Dict, List

# Import necessary classes from the provided datamodel
# Assuming datamodel.py is available in the execution environment
from datamodel import (Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Product, Trade, TradingState)

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

# --- Helper Functions ---
def calculate_mid_price(order_depth: OrderDepth) -> float | None:
    """Calculates the mid-price from the best bid and ask."""
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    best_bid = max(order_depth.buy_orders.keys())
    best_ask = min(order_depth.sell_orders.keys())
    return (best_bid + best_ask) / 2.0

def calculate_sma(prices: List[float], window: int) -> float | None:
    """Calculates the Simple Moving Average."""
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window

def calculate_std_dev(prices: List[float], window: int) -> float | None:
    """Calculates the Standard Deviation."""
    if len(prices) < window:
        return None
    mean = calculate_sma(prices, window)
    if mean is None:
        return None
    variance = sum([(p - mean) ** 2 for p in prices[-window:]]) / window
    return variance ** 0.5

# --- Trader Class ---
class Trader:
    def __init__(self):
        self.position_limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
        }
        # Max length for price history to prevent traderData exceeding limits
        self.max_history_length = 100

        # Parameters (can be tuned)
        self.kelp_sma_short_window = 10
        self.kelp_sma_long_window = 30
        self.kelp_std_dev_window = 20
        self.kelp_volatility_spread_multiplier = 1.5
        self.kelp_base_spread = 1
        self.kelp_base_order_size = 5
        self.kelp_volatility_size_dampening = 0.5 # Reduce size more as volatility increases

        self.component_sma_window = 20
        self.component_std_dev_window = 20
        self.component_spread_multiplier = 1.0
        self.component_base_spread = 1
        self.component_order_size = 10

        self.arbitrage_threshold = 2 # Minimum profit per basket to trigger arbitrage


    # --- Strategy for Rainforest Resin (Keep as provided) ---
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

    # --- New Kelp Strategy ---
    def kelp_strategy_enhanced(self, order_depth: OrderDepth, position: int, trader_state: Dict) -> (List[Order], Dict): # type: ignore
        orders = []
        position_limit = self.position_limits["KELP"]

        # Load historical data from trader_state
        kelp_prices = trader_state.get("kelp_prices", [])
        kelp_sma_short = trader_state.get("kelp_sma_short")
        kelp_sma_long = trader_state.get("kelp_sma_long")
        kelp_std_dev = trader_state.get("kelp_std_dev")

        mid_price = calculate_mid_price(order_depth)
        if mid_price is None:
            return orders, trader_state # Not enough book depth

        # Update history and calculations
        kelp_prices.append(mid_price)
        if len(kelp_prices) > self.max_history_length:
             kelp_prices = kelp_prices[-self.max_history_length:] # Prevent excessive growth

        new_sma_short = calculate_sma(kelp_prices, self.kelp_sma_short_window)
        new_sma_long = calculate_sma(kelp_prices, self.kelp_sma_long_window)
        new_std_dev = calculate_std_dev(kelp_prices, self.kelp_std_dev_window)

        # Update state for next iteration
        trader_state["kelp_prices"] = kelp_prices
        trader_state["kelp_sma_short"] = new_sma_short
        trader_state["kelp_sma_long"] = new_sma_long
        trader_state["kelp_std_dev"] = new_std_dev

        if new_sma_long is None or new_std_dev is None or new_sma_short is None:
             logger.print("KELP: Not enough data for calculations.")
             return orders, trader_state # Need more data

        # Determine Trend Bias
        trend_bias = 0 # Neutral
        if new_sma_short > new_sma_long + 0.5: # Add small buffer
             trend_bias = 1 # Uptrend
        elif new_sma_short < new_sma_long - 0.5:
             trend_bias = -1 # Downtrend

        # Determine Volatility and Adjust Parameters
        volatility = new_std_dev
        dynamic_spread = self.kelp_base_spread + volatility * self.kelp_volatility_spread_multiplier
        # Reduce order size as volatility increases
        dynamic_order_size = max(1, int(self.kelp_base_order_size * (1 - min(0.8, volatility * self.kelp_volatility_size_dampening)))) # Ensure size is at least 1, capped reduction

        # Define Buy/Sell Prices based on Trend Bias
        buy_price = int(mid_price - dynamic_spread / 2)
        sell_price = int(mid_price + dynamic_spread / 2)

        if trend_bias == 1: # Uptrend - bias towards buying closer
            buy_price = int(mid_price - dynamic_spread * 0.4)
            sell_price = int(mid_price + dynamic_spread * 0.6)
        elif trend_bias == -1: # Downtrend - bias towards selling closer
            buy_price = int(mid_price - dynamic_spread * 0.6)
            sell_price = int(mid_price + dynamic_spread * 0.4)

        # Ensure buy price < sell price
        if buy_price >= sell_price:
            buy_price = sell_price - 1

        # Place Orders
        available_buy_capacity = position_limit - position
        available_sell_capacity = position_limit + position

        if available_buy_capacity > 0:
            buy_quantity = min(dynamic_order_size, available_buy_capacity)
            orders.append(Order("KELP", buy_price, buy_quantity))
            logger.print(f"KELP: Placing BUY {buy_quantity}@{buy_price} (Mid: {mid_price:.2f}, Trend: {trend_bias}, Vol: {volatility:.2f}, Size: {dynamic_order_size})")


        if available_sell_capacity > 0:
            sell_quantity = min(dynamic_order_size, available_sell_capacity)
            orders.append(Order("KELP", sell_price, -sell_quantity))
            logger.print(f"KELP: Placing SELL {sell_quantity}@{sell_price} (Mid: {mid_price:.2f}, Trend: {trend_bias}, Vol: {volatility:.2f}, Size: {dynamic_order_size})")

        # Simple Mean Reversion - If price deviates significantly, place small counter-trend order
        if mid_price < new_sma_long - 2 * volatility and available_buy_capacity > 0:
             reversion_qty = min(int(dynamic_order_size * 0.5), available_buy_capacity)
             if reversion_qty > 0:
                 best_ask = min(order_depth.sell_orders.keys())
                 orders.append(Order("KELP", best_ask, reversion_qty)) # Aggressive buy
                 logger.print(f"KELP: Mean Reversion BUY {reversion_qty}@{best_ask}")

        if mid_price > new_sma_long + 2 * volatility and available_sell_capacity > 0:
            reversion_qty = min(int(dynamic_order_size * 0.5), available_sell_capacity)
            if reversion_qty > 0:
                best_bid = max(order_depth.buy_orders.keys())
                orders.append(Order("KELP", best_bid, -reversion_qty)) # Aggressive sell
                logger.print(f"KELP: Mean Reversion SELL {reversion_qty}@{best_bid}")


        return orders, trader_state


    # --- Strategy for Squid Ink (Keep as provided) ---
    def squid_ink_strategy(self, order_depth: OrderDepth, position: int, trader_state: Dict) -> (List[Order], Dict): # type: ignore
        orders = []
        position_limit = self.position_limits["SQUID_INK"]

        # --- Load state from trader_state ---
        squid_price_history = trader_state.get("squid_price_history", [])
        squid_position_history = trader_state.get("squid_position_history", [])
        squid_last_signal = trader_state.get("squid_last_signal", 0)
        squid_z_threshold = trader_state.get("squid_z_threshold", 2.27)
        squid_window_size = trader_state.get("squid_window_size", 20)

        # Check for valid order book presence (Early Exit)
        if not order_depth.sell_orders or not order_depth.buy_orders:
            logger.print("SQUID_INK: Empty order book detected.")
            # Ensure state is saved even on early exit
            trader_state["squid_price_history"] = squid_price_history
            trader_state["squid_position_history"] = squid_position_history
            trader_state["squid_last_signal"] = squid_last_signal
            trader_state["squid_z_threshold"] = squid_z_threshold
            trader_state["squid_window_size"] = squid_window_size
            return orders, trader_state # Return unchanged state

        # Get market prices
        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())
        mid_price = (best_ask + best_bid) // 2 # Use integer division if mid_price should be int

        # Add current price to history (loaded from trader_state)
        squid_price_history.append(mid_price)
        squid_position_history.append(position)

        # Trim history if too long
        if len(squid_price_history) > self.max_history_length:
             squid_price_history = squid_price_history[-self.max_history_length:]
        if len(squid_position_history) > self.max_history_length:
            squid_position_history = squid_position_history[-self.max_history_length:]

        # Need enough data to proceed (Early Exit)
        if len(squid_price_history) < squid_window_size + 5:
            logger.print(f"SQUID_INK: Not enough history ({len(squid_price_history)} < {squid_window_size + 5})")
            # --- Save state back to trader_state before returning ---
            trader_state["squid_price_history"] = squid_price_history
            trader_state["squid_position_history"] = squid_position_history
            trader_state["squid_last_signal"] = squid_last_signal
            trader_state["squid_z_threshold"] = squid_z_threshold
            trader_state["squid_window_size"] = squid_window_size
            return orders, trader_state # Corrected return

        # --- REMOVED REDUNDANT CODE BLOCK (lines 401-418 from original) ---

        # Calculate rolling window statistics (using local variables)
        window = squid_price_history[-squid_window_size:]
        mean_price = sum(window) / len(window)
        variance = sum((p - mean_price) ** 2 for p in window) / len(window)
        std_dev = max(1, (variance) ** 0.5) # Floor of 1 to prevent extreme z-scores

        # Calculate z-score of current price
        z_score = (mid_price - mean_price) / std_dev

        # Position limits
        # position_limit = self.position_limits.get("SQUID_INK", 50) # Already defined above
        buy_capacity = position_limit - position
        sell_capacity = position_limit + position

        # Generate mean reversion signal
        signal = 0
        signal_strength = 0 # Initialize signal_strength

        # Check recent price direction - are we in a strong trend?
        price_trend = 0
        if len(squid_price_history) > 10:
             recent_prices = squid_price_history[-10:]
             price_trend = 1 if recent_prices[-1] > recent_prices[0] else -1

        # Only generate mean reversion signals when z-score is extreme
        # AND when the price is starting to move in the reversal direction
        if z_score < -squid_z_threshold:
            # Price significantly below mean - potential BUY
            # Check if we're seeing signs of upward movement
            if len(squid_price_history) > 3 and squid_price_history[-1] > squid_price_history[-2]:
                signal = 1
                signal_strength = min(1.0, abs(z_score) / 4)
                logger.print(f"SQUID_INK: BUY Signal Triggered (z={z_score:.2f}, strength={signal_strength:.2f})")

        elif z_score > squid_z_threshold:
            # Price significantly above mean - potential SELL
            # Check if we're seeing signs of downward movement
            if len(squid_price_history) > 3 and squid_price_history[-1] < squid_price_history[-2]:
                signal = -1
                signal_strength = min(1.0, abs(z_score) / 4)
                logger.print(f"SQUID_INK: SELL Signal Triggered (z={z_score:.2f}, strength={signal_strength:.2f})")

        # Execution logic - more conservative position sizing
        base_size = 10 # Even smaller base position size
        adjusted_size = base_size # Default size

        # Scale size based on how extreme the z-score is and confirmation signals
        if signal != 0: # Only adjust size if there IS a signal
            if abs(z_score) > 4.0:
                adjusted_size = 12 # Very extreme deviation with confirmation - use larger size
            elif abs(z_score) > 3.5:
                adjusted_size = 8
            elif abs(z_score) > 3.0:
                adjusted_size = 5
            # else: use base_size (already set)

            # Skip trading if liquidity is too thin
            bid_liquidity = sum(order_depth.buy_orders.values())
            ask_liquidity = abs(sum(order_depth.sell_orders.values()))
            min_liquidity_threshold = 1

            if signal == 1 and ask_liquidity < min_liquidity_threshold:
                logger.print(f"SQUID_INK: BUY Signal Skipped (Low Ask Liq: {ask_liquidity})")
                signal = 0 # Not enough sell orders to buy from
            elif signal == -1 and bid_liquidity < min_liquidity_threshold:
                logger.print(f"SQUID_INK: SELL Signal Skipped (Low Bid Liq: {bid_liquidity})")
                signal = 0 # Not enough buy orders to sell to

        # Execute trades based on signal
        if signal == 1 and buy_capacity > 0: # BUY
            # Position size based on available capacity and signal strength
            size = min(
                adjusted_size,
                buy_capacity,
                -order_depth.sell_orders[best_ask] # Available volume at best ask
            )
            if size > 0:
                orders.append(Order("SQUID_INK", best_ask, size))
                logger.print(f"SQUID_INK: Placing BUY Order {size}@{best_ask}")


        elif signal == -1 and sell_capacity > 0: # SELL
             # Position size based on available capacity and signal strength
             size = min(
                 adjusted_size,
                 sell_capacity,
                 order_depth.buy_orders[best_bid] # Available volume at best bid
             )
             if size > 0:
                 orders.append(Order("SQUID_INK", best_bid, -size))
                 logger.print(f"SQUID_INK: Placing SELL Order {size}@{best_bid}")


        # --- Save state back to trader_state before returning ---
        trader_state["squid_price_history"] = squid_price_history
        trader_state["squid_position_history"] = squid_position_history
        trader_state["squid_last_signal"] = signal # Store the actual signal executed (or 0 if skipped/none)
        trader_state["squid_z_threshold"] = squid_z_threshold
        trader_state["squid_window_size"] = squid_window_size

        # Return orders and the updated state (Corrected Final Return)
        return orders, trader_state

    # --- Strategy for Basket Components (Market Making) ---
    def component_market_making(self, product: Product, order_depth: OrderDepth, position: int, trader_state: Dict) -> (List[Order], Dict): # type: ignore
        orders = []
        position_limit = self.position_limits[product]
        state_prefix = product.lower() # e.g., 'croissants_'

        # Load state
        prices = trader_state.get(f"{state_prefix}_prices", [])
        sma = trader_state.get(f"{state_prefix}_sma")
        std_dev = trader_state.get(f"{state_prefix}_std_dev")


        mid_price = calculate_mid_price(order_depth)
        if mid_price is None:
            return orders, trader_state

        # Update history and calculations
        prices.append(mid_price)
        if len(prices) > self.max_history_length:
             prices = prices[-self.max_history_length:]

        new_sma = calculate_sma(prices, self.component_sma_window)
        new_std_dev = calculate_std_dev(prices, self.component_std_dev_window)

        # Update state
        trader_state[f"{state_prefix}_prices"] = prices
        trader_state[f"{state_prefix}_sma"] = new_sma
        trader_state[f"{state_prefix}_std_dev"] = new_std_dev

        if new_sma is None or new_std_dev is None:
            logger.print(f"{product}: Not enough data for market making.")
            return orders, trader_state

        # Basic market making around SMA
        fair_value = new_sma
        spread = max(self.component_base_spread, int(new_std_dev * self.component_spread_multiplier))
        buy_price = int(fair_value - spread)
        sell_price = int(fair_value + spread)

        if buy_price >= sell_price: buy_price = sell_price -1 # Ensure separation

        available_buy_capacity = position_limit - position
        available_sell_capacity = position_limit + position
        order_size = self.component_order_size

        if available_buy_capacity > 0:
            qty = min(order_size, available_buy_capacity)
            orders.append(Order(product, buy_price, qty))

        if available_sell_capacity > 0:
            qty = min(order_size, available_sell_capacity)
            orders.append(Order(product, sell_price, -qty))

        return orders, trader_state


    # --- Strategy for Picnic Basket Arbitrage ---
    def basket_arbitrage_strategy(self, state: TradingState, trader_state: Dict) -> (List[Order], Dict): # type: ignore
        orders = []
        position = state.position
        order_depths = state.order_depths

        # Required products for arbitrage
        required_products = ["PICNIC_BASKET1", "PICNIC_BASKET2", "CROISSANTS", "JAMS", "DJEMBES"]
        if any(prod not in order_depths for prod in required_products):
            logger.print("ARBITRAGE: Missing order depth for a required product.")
            return orders, trader_state # Cannot proceed if any book is missing

        # Get necessary best bids and asks
        prices = {}
        for prod in required_products:
             if not order_depths[prod].buy_orders or not order_depths[prod].sell_orders:
                 logger.print(f"ARBITRAGE: Empty order book for {prod}.")
                 return orders, trader_state # Cannot proceed if any book is empty
             prices[f"{prod}_best_bid"] = max(order_depths[prod].buy_orders.keys())
             prices[f"{prod}_best_ask"] = min(order_depths[prod].sell_orders.keys())
             prices[f"{prod}_mid"] = (prices[f"{prod}_best_bid"] + prices[f"{prod}_best_ask"]) / 2.0


        # --- Basket 1 Arbitrage ---
        # Implied value calculation (using opposite side for conservative estimate)
        # Cost to BUY components = sum(ask_prices * quantity)
        # Value to SELL components = sum(bid_prices * quantity)
        implied_cost_basket1 = (6 * prices["CROISSANTS_best_ask"] +
                                3 * prices["JAMS_best_ask"] +
                                1 * prices["DJEMBES_best_ask"])
        implied_value_basket1 = (6 * prices["CROISSANTS_best_bid"] +
                                 3 * prices["JAMS_best_bid"] +
                                 1 * prices["DJEMBES_best_bid"])

        basket1_bid = prices["PICNIC_BASKET1_best_bid"]
        basket1_ask = prices["PICNIC_BASKET1_best_ask"]

        # Opportunity 1: Buy Basket, Sell Components
        # Profit = Value_Sell_Components - Cost_Buy_Basket
        profit1_buy_basket = implied_value_basket1 - basket1_ask
        if profit1_buy_basket > self.arbitrage_threshold:
            # Calculate max size based on liquidity and position limits
            max_size = float('inf')
            # Basket liquidity (buying basket)
            max_size = min(max_size, -order_depths["PICNIC_BASKET1"].sell_orders.get(basket1_ask, 0))
            # Component liquidity (selling components)
            max_size = min(max_size, order_depths["CROISSANTS"].buy_orders.get(prices["CROISSANTS_best_bid"], 0) // 6)
            max_size = min(max_size, order_depths["JAMS"].buy_orders.get(prices["JAMS_best_bid"], 0) // 3)
            max_size = min(max_size, order_depths["DJEMBES"].buy_orders.get(prices["DJEMBES_best_bid"], 0) // 1)

            # Position limits (Positive means BUY capacity, Negative means SELL capacity)
            # Buying Basket1: need capacity (limit - current_pos)
            max_size = min(max_size, self.position_limits["PICNIC_BASKET1"] - position.get("PICNIC_BASKET1", 0))
             # Selling Croissants: need capacity (limit + current_pos)
            max_size = min(max_size, (self.position_limits["CROISSANTS"] + position.get("CROISSANTS", 0)) // 6)
            max_size = min(max_size, (self.position_limits["JAMS"] + position.get("JAMS", 0)) // 3)
            max_size = min(max_size, (self.position_limits["DJEMBES"] + position.get("DJEMBES", 0)) // 1)


            size = int(max_size)
            if size > 0:
                logger.print(f"ARBITRAGE (BASKET1): Buy Basket @{basket1_ask}, Sell Components. Profit: {profit1_buy_basket:.2f}, Size: {size}")
                orders.append(Order("PICNIC_BASKET1", basket1_ask, size))
                orders.append(Order("CROISSANTS", prices["CROISSANTS_best_bid"], -6 * size))
                orders.append(Order("JAMS", prices["JAMS_best_bid"], -3 * size))
                orders.append(Order("DJEMBES", prices["DJEMBES_best_bid"], -1 * size))


        # Opportunity 2: Sell Basket, Buy Components
        # Profit = Value_Sell_Basket - Cost_Buy_Components
        profit1_sell_basket = basket1_bid - implied_cost_basket1
        if profit1_sell_basket > self.arbitrage_threshold:
             # Calculate max size
            max_size = float('inf')
             # Basket liquidity (selling basket)
            max_size = min(max_size, order_depths["PICNIC_BASKET1"].buy_orders.get(basket1_bid, 0))
             # Component liquidity (buying components)
            max_size = min(max_size, -order_depths["CROISSANTS"].sell_orders.get(prices["CROISSANTS_best_ask"], 0) // 6)
            max_size = min(max_size, -order_depths["JAMS"].sell_orders.get(prices["JAMS_best_ask"], 0) // 3)
            max_size = min(max_size, -order_depths["DJEMBES"].sell_orders.get(prices["DJEMBES_best_ask"], 0) // 1)

             # Position limits
             # Selling Basket1: need capacity (limit + current_pos)
            max_size = min(max_size, self.position_limits["PICNIC_BASKET1"] + position.get("PICNIC_BASKET1", 0))
             # Buying Croissants: need capacity (limit - current_pos)
            max_size = min(max_size, (self.position_limits["CROISSANTS"] - position.get("CROISSANTS", 0)) // 6)
            max_size = min(max_size, (self.position_limits["JAMS"] - position.get("JAMS", 0)) // 3)
            max_size = min(max_size, (self.position_limits["DJEMBES"] - position.get("DJEMBES", 0)) // 1)

            size = int(max_size)
            if size > 0:
                logger.print(f"ARBITRAGE (BASKET1): Sell Basket @{basket1_bid}, Buy Components. Profit: {profit1_sell_basket:.2f}, Size: {size}")
                orders.append(Order("PICNIC_BASKET1", basket1_bid, -size))
                orders.append(Order("CROISSANTS", prices["CROISSANTS_best_ask"], 6 * size))
                orders.append(Order("JAMS", prices["JAMS_best_ask"], 3 * size))
                orders.append(Order("DJEMBES", prices["DJEMBES_best_ask"], 1 * size))


        # --- Basket 2 Arbitrage (Similar logic) ---
        implied_cost_basket2 = (4 * prices["CROISSANTS_best_ask"] +
                                2 * prices["JAMS_best_ask"])
        implied_value_basket2 = (4 * prices["CROISSANTS_best_bid"] +
                                 2 * prices["JAMS_best_bid"])

        basket2_bid = prices["PICNIC_BASKET2_best_bid"]
        basket2_ask = prices["PICNIC_BASKET2_best_ask"]

        # Opportunity 1: Buy Basket, Sell Components
        profit2_buy_basket = implied_value_basket2 - basket2_ask
        if profit2_buy_basket > self.arbitrage_threshold:
            max_size = float('inf')
            max_size = min(max_size, -order_depths["PICNIC_BASKET2"].sell_orders.get(basket2_ask, 0))
            max_size = min(max_size, order_depths["CROISSANTS"].buy_orders.get(prices["CROISSANTS_best_bid"], 0) // 4)
            max_size = min(max_size, order_depths["JAMS"].buy_orders.get(prices["JAMS_best_bid"], 0) // 2)

            max_size = min(max_size, self.position_limits["PICNIC_BASKET2"] - position.get("PICNIC_BASKET2", 0))
            max_size = min(max_size, (self.position_limits["CROISSANTS"] + position.get("CROISSANTS", 0)) // 4)
            max_size = min(max_size, (self.position_limits["JAMS"] + position.get("JAMS", 0)) // 2)

            size = int(max_size)
            if size > 0:
                logger.print(f"ARBITRAGE (BASKET2): Buy Basket @{basket2_ask}, Sell Components. Profit: {profit2_buy_basket:.2f}, Size: {size}")
                orders.append(Order("PICNIC_BASKET2", basket2_ask, size))
                orders.append(Order("CROISSANTS", prices["CROISSANTS_best_bid"], -4 * size))
                orders.append(Order("JAMS", prices["JAMS_best_bid"], -2 * size))

        # Opportunity 2: Sell Basket, Buy Components
        profit2_sell_basket = basket2_bid - implied_cost_basket2
        if profit2_sell_basket > self.arbitrage_threshold:
            max_size = float('inf')
            max_size = min(max_size, order_depths["PICNIC_BASKET2"].buy_orders.get(basket2_bid, 0))
            max_size = min(max_size, -order_depths["CROISSANTS"].sell_orders.get(prices["CROISSANTS_best_ask"], 0) // 4)
            max_size = min(max_size, -order_depths["JAMS"].sell_orders.get(prices["JAMS_best_ask"], 0) // 2)

            max_size = min(max_size, self.position_limits["PICNIC_BASKET2"] + position.get("PICNIC_BASKET2", 0))
            max_size = min(max_size, (self.position_limits["CROISSANTS"] - position.get("CROISSANTS", 0)) // 4)
            max_size = min(max_size, (self.position_limits["JAMS"] - position.get("JAMS", 0)) // 2)

            size = int(max_size)
            if size > 0:
                logger.print(f"ARBITRAGE (BASKET2): Sell Basket @{basket2_bid}, Buy Components. Profit: {profit2_sell_basket:.2f}, Size: {size}")
                orders.append(Order("PICNIC_BASKET2", basket2_bid, -size))
                orders.append(Order("CROISSANTS", prices["CROISSANTS_best_ask"], 4 * size))
                orders.append(Order("JAMS", prices["JAMS_best_ask"], 2 * size))

        return orders, trader_state


    # --- Main Run Method ---
    def run(self, state: TradingState):
        result = {} # Stores orders for all products
        conversions = 0 # No conversions used in these strategies

        # Load trader state data
        try:
            trader_state = json.loads(state.traderData) if state.traderData else {}
        except json.JSONDecodeError:
            trader_state = {}
            logger.print("Error decoding traderData, starting fresh state.")


        # --- Arbitrage Strategy (Run First) ---
        # This generates potential orders for baskets and components based on arbitrage
        arbitrage_orders, trader_state = self.basket_arbitrage_strategy(state, trader_state)
        for order in arbitrage_orders:
            if order.symbol not in result:
                result[order.symbol] = []
            result[order.symbol].append(order)


        # --- Individual Product Strategies ---
        products_to_trade = [
            "RAINFOREST_RESIN", "KELP", "SQUID_INK",
            "CROISSANTS", "JAMS", "DJEMBES" # Add individual components for market making
            # Baskets are primarily handled by arbitrage, could add MM here too if needed
        ]

        for product in products_to_trade:
            if product not in state.order_depths:
                continue # Skip if no order depth data

            # Skip component/basket trading if already handled by arbitrage orders for this iteration
            # This prevents conflicting orders (e.g. arbitrage sells croissants, MM tries to buy)
            # Note: This is a simple prevention, more complex logic might be needed
            if product in result:
                logger.print(f"Skipping independent strategy for {product} due to existing arbitrage orders.")
                continue

            position = state.position.get(product, 0)
            order_depth = state.order_depths[product]
            orders = [] # Orders for this specific product's strategy

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
                orders, trader_state = self.kelp_strategy_enhanced(order_depth, position, trader_state)

            elif product == "SQUID_INK":
                 orders, trader_state = self.squid_ink_strategy(order_depth, position, trader_state)

            elif product in ["CROISSANTS", "JAMS", "DJEMBES"]:
                 orders, trader_state = self.component_market_making(product, order_depth, position, trader_state)

            # Add generated orders to the result dictionary
            if orders:
                if product not in result:
                     result[product] = []
                result[product].extend(orders) # Use extend to add multiple orders

        # Save updated state back to traderData
        # Ensure data types are JSON serializable (lists, dicts, numbers, strings)
        serializable_trader_state = {}
        for key, value in trader_state.items():
            if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                serializable_trader_state[key] = value
            else:
                 # Handle or log non-serializable types if necessary
                 logger.print(f"Warning: Skipping non-serializable type in traderData: {key} ({type(value)})")


        traderData = json.dumps(serializable_trader_state)

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

