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

#helper functions
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

logger = Logger()

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    SPREAD = "SPREAD"
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    VOLCANIC_ROCK_VOUCHER_9500 = "VOLCANIC_ROCK_VOUCHER_9500"
    VOLCANIC_ROCK_VOUCHER_9750 = "VOLCANIC_ROCK_VOUCHER_9750"
    VOLCANIC_ROCK_VOUCHER_10000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VOLCANIC_ROCK_VOUCHER_10250 = "VOLCANIC_ROCK_VOUCHER_10250"
    VOLCANIC_ROCK_VOUCHER_10500 = "VOLCANIC_ROCK_VOUCHER_10500"


PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 25,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": -0.25,
        "prevent_adverse": False,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": -0.25,
        "prevent_adverse": False,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            'VOLCANIC_ROCK': 400,
            'VOLCANIC_ROCK_VOUCHER_9500': 200,
            'VOLCANIC_ROCK_VOUCHER_9750': 200,
            'VOLCANIC_ROCK_VOUCHER_10000': 200,
            'VOLCANIC_ROCK_VOUCHER_10250': 200,
            'VOLCANIC_ROCK_VOUCHER_10500': 200
        }

        self.position_limits = {
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
        }

        # Max length for price history to prevent traderData exceeding limits
        self.max_history_length = 100

        self.component_sma_window = 20
        self.component_std_dev_window = 20
        self.component_spread_multiplier = 1.0
        self.component_base_spread = 1
        self.component_order_size = 10

        self.arbitrage_threshold = 2 # Minimum profit per basket to trigger arbitrage

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
            
            if traderObject.get(f"{product}_last_price", None) is not None:
                last_price = traderObject[f"{product}_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * self.params[product]["reversion_beta"]
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
        
        # Use a loop instead of recursion
        while True:
            orders_processed = False
            
            # Process sell orders (buying opportunities)
            sell_prices = sorted(order_depth.sell_orders.keys())  # Sort by price (ascending)
            for ask_price in sell_prices:
                if ask_price > fair_value - take_width:
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
                if bid_price < fair_value + take_width:
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
    ) -> (int, int): # type: ignore
        
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
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
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

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

    def KELP_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("KELP_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["KELP_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("KELP_last_price", None) != None:
                last_price = traderObject["KELP_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["KELP_last_price"] = mmmid_price
            return fair
        return None
    
    def SQUID_INK_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            # Complete the rest of the function as per KELP_fair_value
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            
            if mm_ask == None or mm_bid == None:
                if traderObject.get("SQUID_INK_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["SQUID_INK_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
                
            if traderObject.get("SQUID_INK_last_price", None) != None:
                last_price = traderObject["SQUID_INK_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.SQUID_INK]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
                
            traderObject["SQUID_INK_last_price"] = mmmid_price
            return fair
        return None

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int): # type: ignore
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
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ):
        orders: List[Order] = []
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

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  
            else:
                ask = best_ask_above_fair - 1  

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
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
                # orders.append(Order("CROISSANTS", prices["CROISSANTS_best_bid"], -6 * size))
                # orders.append(Order("JAMS", prices["JAMS_best_bid"], -3 * size))
                # orders.append(Order("DJEMBES", prices["DJEMBES_best_bid"], -1 * size))


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
                # orders.append(Order("CROISSANTS", prices["CROISSANTS_best_ask"], 6 * size))
                # orders.append(Order("JAMS", prices["JAMS_best_ask"], 3 * size))
                # orders.append(Order("DJEMBES", prices["DJEMBES_best_ask"], 1 * size))


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
                # orders.append(Order("CROISSANTS", prices["CROISSANTS_best_bid"], -4 * size))
                # orders.append(Order("JAMS", prices["JAMS_best_bid"], -2 * size))

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
                # orders.append(Order("CROISSANTS", prices["CROISSANTS_best_ask"], 4 * size))
                # orders.append(Order("JAMS", prices["JAMS_best_ask"], 2 * size))

        return orders, trader_state   
     
    def black_scholes_call(self, S, K, T, r, sigma):
        """Calculate Black-Scholes price for a call option."""
        if T <= 0:
            return max(0, S - K)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    def implied_volatility(self, price, S, K, T, r):
        """Calculate implied volatility using the Black-Scholes model."""
        if price <= 0 or T <= 0 or S <= 0 or K <= 0:
            return None
            
        # Define the objective function (difference between market and model price)
        def objective(sigma):
            return self.black_scholes_call(S, K, T, r, sigma) - price
            
        try:
            # Use brentq method to find the root (implied volatility)
            return brentq(objective, 0.001, 5.0, maxiter=100)
        except Exception as e:
            logger.print(f"Error calculating implied volatility: {str(e)}")
            return None

    def fit_volatility_smile(self, moneyness_list, iv_list):
        """Fit a parabolic curve to the volatility smile."""
        if len(moneyness_list) < 3:
            return None
        
        # Use numpy's polyfit to fit a quadratic function (parabola)
        coeffs = np.polyfit(moneyness_list, iv_list, 2)
        return coeffs
        
    def evaluate_volatility_smile(self, coeffs, moneyness):
        """Evaluate the fitted volatility smile at a given moneyness."""
        if coeffs is None:
            return None
        
        # Evaluate the polynomial at the given moneyness
        return coeffs[0] * moneyness**2 + coeffs[1] * moneyness + coeffs[2]
    
    def volatility_scalping_strategy(self, state):
        """
        Implements a volatility scalping algorithm where we sell 1% of holdings for
        every 1.5% price increase and buy back 1.0125% with every 1.5% price decrease.
        """
        result = {}
        
        # Define voucher symbols
        voucher_symbols = [
            Product.VOLCANIC_ROCK_VOUCHER_9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500
        ]
        
        # Get volcanic rock price if available
        volcanic_rock_price = None
        if Product.VOLCANIC_ROCK in state.order_depths:
            vr_depth = state.order_depths[Product.VOLCANIC_ROCK]
            if vr_depth.buy_orders and vr_depth.sell_orders:
                best_bid = max(vr_depth.buy_orders.keys())
                best_ask = min(vr_depth.sell_orders.keys())
                volcanic_rock_price = (best_bid + best_ask) / 2
                logger.print(f"VOLCANIC_ROCK price: {volcanic_rock_price}")
        
        if volcanic_rock_price is None:
            return result
            
        # Process each voucher
        for symbol in voucher_symbols:
            if symbol not in state.order_depths:
                continue
                
            depth = state.order_depths[symbol]
            if not depth.buy_orders and not depth.sell_orders:
                continue
                
            position = state.position.get(symbol, 0)
            
            # Calculate mid price
            if depth.buy_orders and depth.sell_orders:
                best_bid = max(depth.buy_orders.keys())
                best_ask = min(depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
                
                # Implement volatility scalping algorithm
                orders = []
                
                # If we have a position and there are buy orders
                if position > 0 and depth.buy_orders:
                    # Calculate how much to sell (1% of holdings)
                    sell_quantity = max(1, int(position * 0.01))
                    if sell_quantity > 0 and position - sell_quantity >= -200:
                        orders.append(Order(symbol, best_bid, -sell_quantity))
                        logger.print(f"VOLATILITY SELL: {sell_quantity} {symbol} @ {best_bid}")
                
                # If we have room to buy and there are sell orders
                if position < 200 and depth.sell_orders:
                    # Calculate how much to buy (1.0125% of potential holdings)
                    potential_holdings = 200 - position
                    buy_quantity = max(1, int(potential_holdings * 0.010125))
                    if buy_quantity > 0:
                        orders.append(Order(symbol, best_ask, buy_quantity))
                        logger.print(f"VOLATILITY BUY: {buy_quantity} {symbol} @ {best_ask}")
                
                if orders:
                    result[symbol] = orders
                    
        return result
    
    def trade_volcanic_rock(self, state):
        """
        Implements a market-making strategy for VOLCANIC_ROCK.
        Computes the mid-price from the order book and places buy and sell orders
        with a small offset. Adjusts order size to remain within the 400 unit position limit.
        """
        orders = []
        
        if Product.VOLCANIC_ROCK not in state.order_depths:
            return orders
        
        depth = state.order_depths[Product.VOLCANIC_ROCK]
        current_position = state.position.get(Product.VOLCANIC_ROCK, 0)
        
        # Determine mid-price based on available buy and sell orders.
        if depth.buy_orders and depth.sell_orders:
            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
        elif depth.sell_orders:
            mid_price = min(depth.sell_orders.keys())
        elif depth.buy_orders:
            mid_price = max(depth.buy_orders.keys())
        else:
            return orders
        
        # Use a fixed offset to place orders around the mid price.
        offset = 5  # You can adjust this offset based on backtesting results
        
        # Place a buy order if we haven't hit the buying limit.
        if current_position < self.LIMIT[Product.VOLCANIC_ROCK]:
            # For example, buy a maximum of 10 units at mid_price - offset
            buy_qty = min(10, self.LIMIT[Product.VOLCANIC_ROCK] - current_position)
            orders.append(Order(Product.VOLCANIC_ROCK, mid_price - offset, buy_qty))
            logger.print(f"Created BUY order: {buy_qty} VOLCANIC_ROCK @ {mid_price - offset}")
        
        # Place a sell order if we haven't hit the selling limit.
        if current_position > -self.LIMIT[Product.VOLCANIC_ROCK]:
            # For example, sell (negative order quantity) a maximum of 10 units at mid_price + offset
            sell_qty = min(10, self.LIMIT[Product.VOLCANIC_ROCK] + current_position)
            orders.append(Order(Product.VOLCANIC_ROCK, mid_price + offset, -sell_qty))
            logger.print(f"Created SELL order: {sell_qty} VOLCANIC_ROCK @ {mid_price + offset}")
        
        return orders
    
    def volcanic_rock_strategy(self, state, risk_free_rate=0.01):
        """Implement trading strategy for Volcanic Rock Vouchers."""
        # Check if Volcanic Rock is available in the market
        if 'VOLCANIC_ROCK' not in state.order_depths:
            return {}
        
        # Get the current price of Volcanic Rock
        volcanic_rock_depth = state.order_depths['VOLCANIC_ROCK']
        if not volcanic_rock_depth.buy_orders or not volcanic_rock_depth.sell_orders:
            return {}
        
        # Calculate mid price of Volcanic Rock
        best_bid = max(volcanic_rock_depth.buy_orders.keys())
        best_ask = min(volcanic_rock_depth.sell_orders.keys())
        volcanic_rock_price = (best_bid + best_ask) / 2
        
        # Calculate days to expiration (assuming we know the current day)
        # In a real implementation, we would track this based on state.timestamp
        days_to_expiration = 7 - (state.timestamp // 1000000)  # Rough estimate
        if days_to_expiration <= 0:
            days_to_expiration = 0.01  # Small positive value to avoid division by zero
        
        # Time to expiration in years (assuming 252 trading days per year)
        T = days_to_expiration / 252
        
        # Collect data for volatility smile fitting
        moneyness_list = []
        iv_list = []
        
        # Dictionary to store orders
        voucher_orders = {}
        
        # Process each voucher
        voucher_symbols = [
            'VOLCANIC_ROCK_VOUCHER_9500',
            'VOLCANIC_ROCK_VOUCHER_9750',
            'VOLCANIC_ROCK_VOUCHER_10000',
            'VOLCANIC_ROCK_VOUCHER_10250',
            'VOLCANIC_ROCK_VOUCHER_10500'
        ]
        
        strike_prices = {
            'VOLCANIC_ROCK_VOUCHER_9500': 9500,
            'VOLCANIC_ROCK_VOUCHER_9750': 9750,
            'VOLCANIC_ROCK_VOUCHER_10000': 10000,
            'VOLCANIC_ROCK_VOUCHER_10250': 10250,
            'VOLCANIC_ROCK_VOUCHER_10500': 10500
        }
        
        # Calculate implied volatilities for each voucher
        for symbol in voucher_symbols:
            if symbol not in state.order_depths:
                continue
                
            voucher_depth = state.order_depths[symbol]
            if not voucher_depth.buy_orders or not voucher_depth.sell_orders:
                continue
                
            # Calculate mid price of the voucher
            voucher_bid = max(voucher_depth.buy_orders.keys())
            voucher_ask = min(voucher_depth.sell_orders.keys())
            voucher_price = (voucher_bid + voucher_ask) / 2
            
            # Calculate moneyness
            strike = strike_prices[symbol]
            moneyness = np.log(strike / volcanic_rock_price) / np.sqrt(T)
            
            # Calculate implied volatility
            iv = self.implied_volatility(voucher_price, volcanic_rock_price, strike, T, risk_free_rate)
            
            if iv is not None:
                moneyness_list.append(moneyness)
                iv_list.append(iv)
        
        # Fit volatility smile if we have enough data points
        if len(moneyness_list) >= 3:
            smile_coeffs = self.fit_volatility_smile(moneyness_list, iv_list)
            
            # Trade based on the fitted volatility smile
            for symbol in voucher_symbols:
                if symbol not in state.order_depths:
                    continue
                    
                voucher_depth = state.order_depths[symbol]
                if not voucher_depth.buy_orders or not voucher_depth.sell_orders:
                    continue
                    
                # Get current position
                position = state.position.get(symbol, 0)
                
                # Calculate theoretical price using the fitted volatility smile
                strike = strike_prices[symbol]
                moneyness = np.log(strike / volcanic_rock_price) / np.sqrt(T)
                theoretical_iv = self.evaluate_volatility_smile(smile_coeffs, moneyness)
                
                if theoretical_iv is None:
                    continue
                    
                theoretical_price = self.black_scholes_call(
                    volcanic_rock_price, strike, T, risk_free_rate, theoretical_iv
                )
                
                # Trading logic based on the difference between theoretical and market price
                best_bid = max(voucher_depth.buy_orders.keys())
                best_ask = min(voucher_depth.sell_orders.keys())
                
                # Define trading thresholds
                bid_threshold = theoretical_price * 0.98  # 2% below theoretical
                ask_threshold = theoretical_price * 1.02  # 2% above theoretical
                
                orders = []
                
                # If market bid is higher than our ask threshold, sell
                if best_bid > ask_threshold and position > -200:
                    quantity = min(voucher_depth.buy_orders[best_bid], 200 + position)
                    if quantity > 0:
                        orders.append(Order(symbol, best_bid, -quantity))
                
                # If market ask is lower than our bid threshold, buy
                if best_ask < bid_threshold and position < 200:
                    quantity = min(abs(voucher_depth.sell_orders[best_ask]), 200 - position)
                    if quantity > 0:
                        orders.append(Order(symbol, best_ask, quantity))
                
                if orders:
                    voucher_orders[symbol] = orders
        
        return voucher_orders

    def analyze_base_iv(self, state, traderObject):
        """Analyze the base IV (at-the-money implied volatility) over time."""
        # Calculate current base IV
        if 'VOLCANIC_ROCK' not in state.order_depths:
            return
        
        volcanic_rock_depth = state.order_depths['VOLCANIC_ROCK']
        if not volcanic_rock_depth.buy_orders or not volcanic_rock_depth.sell_orders:
            return
        
        # Calculate mid price of Volcanic Rock
        best_bid = max(volcanic_rock_depth.buy_orders.keys())
        best_ask = min(volcanic_rock_depth.sell_orders.keys())
        volcanic_rock_price = (best_bid + best_ask) / 2
        
        # Calculate days to expiration
        days_to_expiration = 7 - (state.timestamp // 1000000)  # Rough estimate
        if days_to_expiration <= 0:
            days_to_expiration = 0.01
        
        # Time to expiration in years
        T = days_to_expiration / 252
        
        # Find the voucher with strike closest to current price
        closest_voucher = None
        min_distance = float('inf')
        
        voucher_symbols = [
            'VOLCANIC_ROCK_VOUCHER_9500',
            'VOLCANIC_ROCK_VOUCHER_9750',
            'VOLCANIC_ROCK_VOUCHER_10000',
            'VOLCANIC_ROCK_VOUCHER_10250',
            'VOLCANIC_ROCK_VOUCHER_10500'
        ]
        
        strike_prices = {
            'VOLCANIC_ROCK_VOUCHER_9500': 9500,
            'VOLCANIC_ROCK_VOUCHER_9750': 9750,
            'VOLCANIC_ROCK_VOUCHER_10000': 10000,
            'VOLCANIC_ROCK_VOUCHER_10250': 10250,
            'VOLCANIC_ROCK_VOUCHER_10500': 10500
        }
        
        for symbol in voucher_symbols:
            if symbol not in state.order_depths:
                continue
            
            strike = strike_prices[symbol]
            distance = abs(strike - volcanic_rock_price)
            
            if distance < min_distance:
                min_distance = distance
                closest_voucher = symbol
        
        if closest_voucher is None:
            return
        
        # Calculate IV for the closest voucher
        voucher_depth = state.order_depths[closest_voucher]
        if not voucher_depth.buy_orders or not voucher_depth.sell_orders:
            return
        
        voucher_bid = max(voucher_depth.buy_orders.keys())
        voucher_ask = min(voucher_depth.sell_orders.keys())
        voucher_price = (voucher_bid + voucher_ask) / 2
        
        strike = strike_prices[closest_voucher]
        iv = self.implied_volatility(voucher_price, volcanic_rock_price, strike, T, 0.01)
        
        if iv is None:
            return
        
        # Store base IV history
        if 'base_iv_history' not in traderObject:
            traderObject['base_iv_history'] = []
        
        traderObject['base_iv_history'].append(iv)
        
        # Analyze base IV trend if we have enough data points
        if len(traderObject['base_iv_history']) >= 5:
            recent_ivs = traderObject['base_iv_history'][-5:]
            avg_iv = sum(recent_ivs) / 5
            
            # Check if current IV is significantly higher or lower than average
            if iv > avg_iv * 1.1:  # IV is 10% higher than average
                traderObject['iv_trend'] = 'high'
            elif iv < avg_iv * 0.9:  # IV is 10% lower than average
                traderObject['iv_trend'] = 'low'
            else:
                traderObject['iv_trend'] = 'neutral'
        
        return traderObject

    def run(self, state: TradingState):
        # Define voucher symbols
        voucher_symbols = [
            Product.VOLCANIC_ROCK_VOUCHER_9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500
        ]
        
        strike_prices = {
            Product.VOLCANIC_ROCK_VOUCHER_9500: 9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750: 9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000: 10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250: 10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500: 10500
        }

        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        
        result = {}
        conversions = 0
        trader_data = ""
        
        # Initialize orders as an empty list before using it
        orders = []
        
        # Add Round 1 products trading
        for product in [Product.RAINFOREST_RESIN, Product.KELP, Product.SQUID_INK]:
            if product in state.order_depths:
                position = state.position.get(product, 0)
                # Initialize buy_order_volume and sell_order_volume
                buy_order_volume = 0
                sell_order_volume = 0
                
                # Calculate fair value based on product type
                if product == Product.RAINFOREST_RESIN:
                    fair_value = self.params[product]["fair_value"]
                elif product == Product.KELP:
                    fair_value = self.KELP_fair_value(state.order_depths[product], traderObject)
                elif product == Product.SQUID_INK:
                    fair_value = self.SQUID_INK_fair_value(state.order_depths[product], traderObject)
                    
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
                        product == Product.RAINFOREST_RESIN,
                        self.params[product].get("soft_position_limit", 0),
                    )
                    
                    result[product] = take_orders + clear_orders + make_orders
        
        # New individual products (CROISSANTS, JAMS, DJEMBES)
        for product in [Product.CROISSANTS, Product.JAMS, Product.DJEMBES]:
            if product in self.params and product in state.order_depths:
                position = state.position.get(product, 0)
                
                # Calculate fair value (similar to KELP/SQUID_INK)
                fair_value = self.calculate_fair_value(product, state.order_depths[product], traderObject)
                
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
                )
                
                result[product] = take_orders + clear_orders + make_orders
        
        # Basket products (PICNIC_BASKET1, PICNIC_BASKET2)
        # Before calling basket_arbitrage_strategy, initialize trader_state
        trader_state = {} if state.traderData == "" else json.loads(state.traderData)
        arbitrage_orders, trader_state = self.basket_arbitrage_strategy(state, trader_state)

        for order in arbitrage_orders:
            if order.symbol not in result:
                result[order.symbol] = []
            result[order.symbol].append(order)
        
        # First, get the current price of the underlying VOLCANIC_ROCK
        volcanic_rock_price = None
        if Product.VOLCANIC_ROCK in state.order_depths:
            vr_depth = state.order_depths[Product.VOLCANIC_ROCK]
            if vr_depth.buy_orders and vr_depth.sell_orders:
                best_bid = max(vr_depth.buy_orders.keys())
                best_ask = min(vr_depth.sell_orders.keys())
                volcanic_rock_price = (best_bid + best_ask) / 2
                logger.print(f"Current VOLCANIC_ROCK price: {volcanic_rock_price}")
        
        # Trade VOLCANIC_ROCK with a fixed spread strategy
        if Product.VOLCANIC_ROCK in state.order_depths:
            rock_depth = state.order_depths[Product.VOLCANIC_ROCK]
            position = state.position.get(Product.VOLCANIC_ROCK, 0)
            
            if rock_depth.buy_orders and rock_depth.sell_orders:
                best_bid = max(rock_depth.buy_orders.keys())
                best_ask = min(rock_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
                
                # Define a reasonable spread
                spread = 10
                our_bid = mid_price - spread/2
                our_ask = mid_price + spread/2
                
                rock_orders = []
                
                # Place buy order only if our position allows and our bid price makes sense
                if position < self.LIMIT[Product.VOLCANIC_ROCK] and our_bid < best_ask:
                    buy_quantity = min(10, self.LIMIT[Product.VOLCANIC_ROCK] - position)
                    if buy_quantity > 0:
                        rock_orders.append(Order(Product.VOLCANIC_ROCK, int(our_bid), buy_quantity))
                
                # Place sell order only if our position allows and our ask price makes sense
                if position > -self.LIMIT[Product.VOLCANIC_ROCK] and our_ask > best_bid:
                    sell_quantity = min(10, self.LIMIT[Product.VOLCANIC_ROCK] + position)
                    if sell_quantity > 0:
                        rock_orders.append(Order(Product.VOLCANIC_ROCK, int(our_ask), -sell_quantity))
                        
                if rock_orders:
                    result[Product.VOLCANIC_ROCK] = rock_orders
                    
        # Estimate days to expiration (in a real scenario, this would be tracked properly)
        days_to_expiration = max(1, 7 - (state.timestamp // 1000000))
        T = days_to_expiration / 252  # Time in years, assuming 252 trading days
        risk_free_rate = 0.01
        
        # Trade each voucher using option pricing logic
        for symbol in voucher_symbols:
            if symbol not in state.order_depths:
                continue
                
            depth = state.order_depths[symbol]
            if not depth.buy_orders or not depth.sell_orders:
                continue
                
            position = state.position.get(symbol, 0)
            logger.print(f"Current position for {symbol}: {position}")
            
            best_bid = max(depth.buy_orders.keys())
            best_ask = min(depth.sell_orders.keys())
            market_mid = (best_bid + best_ask) / 2
            
            # Skip if we don't have the underlying price
            if volcanic_rock_price is None:
                continue
                
            # Calculate theoretical price using a simple model
            # For now, we'll use a very basic approximation
            strike = strike_prices[symbol]
            moneyness = volcanic_rock_price / strike
            
            # This is a simplified model, not Black-Scholes
            if volcanic_rock_price > strike:
                # In-the-money option
                intrinsic_value = volcanic_rock_price - strike
                time_value = strike * 0.03 * T  # Simple time value estimate
                theoretical_price = max(1, intrinsic_value + time_value)
            else:
                # Out-of-the-money option
                intrinsic_value = 0
                time_value = strike * 0.02 * T * max(0.01, 1 - (strike - volcanic_rock_price) / strike)
                theoretical_price = max(1, time_value)
                        
            # Trading logic - only trade if there's a significant mispricing
            symbol_orders = []
            mispricing_threshold = 0.05  # 5% threshold
            
            # Only buy if the market price is significantly below our theoretical price
            if market_mid < theoretical_price * (1 - mispricing_threshold) and position < 200:
                # Market price is too low, we should buy
                volume_to_buy = min(abs(depth.sell_orders[best_ask]), 200 - position)
                if volume_to_buy > 0:
                    symbol_orders.append(Order(symbol, best_ask, volume_to_buy))
            
            # Only sell if the market price is significantly above our theoretical price
            elif market_mid > theoretical_price * (1 + mispricing_threshold) and position > -200:
                # Market price is too high, we should sell
                volume_to_sell = min(depth.buy_orders[best_bid], 200 + position)
                if volume_to_sell > 0:
                    symbol_orders.append(Order(symbol, best_bid, -volume_to_sell))

            # Add orders to result if any were created
            if symbol_orders:
                result[symbol] = symbol_orders
                
        # Store updated trader data
        trader_data = jsonpickle.encode(traderObject)
        
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data