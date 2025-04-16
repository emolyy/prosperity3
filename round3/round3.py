
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any, Dict
import jsonpickle
import numpy as np
import json
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

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

BASKETS_PRODUCTS = {
    Product.PICNIC_BASKET1: 
    {
        Product.CROISSANTS : 6,
        Product.JAMS : 3,
        Product.DJEMBES : 1
    },
    Product.PICNIC_BASKET2:
    {
        Product.CROISSANTS : 4,
        Product.JAMS : 2
    }
}

BASKET1_PRODS = {
    Product.CROISSANTS : 6,
    Product.JAMS : 3,
    Product.DJEMBES : 1
}

BASKET2_PRODS = {
    Product.CROISSANTS : 4,
    Product.JAMS : 2
}

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
    Product.SPREAD:{
        "spread_mean": 379.50439988,
        "starting_its": 30000,
        "spread_std_window": 25,
        "zscore_threshold": 11,
        "target_position": 60,
    },
    # Add to the PARAMS dictionary
    Product.CROISSANTS: {
        "take_width": 1.5,  # Slightly wider to catch more opportunities
        "clear_width": -0.5,  # More aggressive clearing
        "prevent_adverse": True,  # Prevent adverse selection
        "adverse_volume": 25,  # Higher threshold for adverse selection
        "reversion_beta": -0.3,  # Stronger mean reversion coefficient
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 2,  # Slightly higher edge
        "position_limit": 200,  # Use more of the position limit
    },
    Product.JAMS: {
        "take_width": 1,
        "clear_width": -0.25,
        "prevent_adverse": False,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.DJEMBES: {
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
    ) -> (int, int): # type: ignore
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

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

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int): # type: ignore
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

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

    def basket_arbitrage(self, state, product, basket_components, result):
        basket_position = state.position.get(product, 0)
        
        # Calculate theoretical basket value
        theoretical_value = 0
        component_availability = {}
        
        for component, quantity in basket_components.items():
            if component not in state.order_depths:
                return  # Skip if any component is missing
            
            component_depth = state.order_depths[component]
            if not component_depth.sell_orders:
                return  # Skip if we can't buy the component
                
            best_ask = min(component_depth.sell_orders.keys())
            theoretical_value += best_ask * quantity
            
            # Check if we have enough available to sell
            component_position = state.position.get(component, 0)
            component_availability[component] = component_position >= quantity
        
        # Check if basket is in the order book
        if product not in state.order_depths:
            return
            
        basket_depth = state.order_depths[product]
        
        # Arbitrage opportunity: Buy basket, sell components
        if basket_depth.sell_orders and basket_position < self.LIMIT[product]:
            best_basket_ask = min(basket_depth.sell_orders.keys())
            if best_basket_ask < theoretical_value:
                # Buy the basket
                basket_quantity = min(abs(basket_depth.sell_orders[best_basket_ask]), 
                                    self.LIMIT[product] - basket_position)
                if basket_quantity > 0:
                    result.setdefault(product, []).append(Order(product, best_basket_ask, basket_quantity))
                    
                    # Sell the components
                    for component, quantity in basket_components.items():
                        component_depth = state.order_depths[component]
                        if component_depth.buy_orders:
                            best_bid = max(component_depth.buy_orders.keys())
                            component_quantity = quantity * basket_quantity
                            result.setdefault(component, []).append(Order(component, best_bid, -component_quantity))
        
        # Arbitrage opportunity: Buy components, sell basket
        all_components_available = all(component_availability.values())
        if basket_depth.buy_orders and all_components_available and basket_position > -self.LIMIT[product]:
            best_basket_bid = max(basket_depth.buy_orders.keys())
            if best_basket_bid > theoretical_value:
                # Sell the basket
                basket_quantity = min(basket_depth.buy_orders[best_basket_bid], 
                                    self.LIMIT[product] + basket_position)
                if basket_quantity > 0:
                    result.setdefault(product, []).append(Order(product, best_basket_bid, -basket_quantity))
                    
                    # Buy the components
                    for component, quantity in basket_components.items():
                        component_depth = state.order_depths[component]
                        if component_depth.sell_orders:
                            best_ask = min(component_depth.sell_orders.keys())
                            component_quantity = quantity * basket_quantity
                            result.setdefault(component, []).append(Order(component, best_ask, component_quantity))

    def basket_stat_arb(self, state, product, basket_components, result):
        basket_position = state.position.get(product, 0)
        
        # Calculate the spread between basket price and sum of components
        if product not in state.order_depths:
            return
            
        basket_depth = state.order_depths[product]
        if not basket_depth.buy_orders or not basket_depth.sell_orders:
            return
            
        basket_mid = (max(basket_depth.buy_orders.keys()) + min(basket_depth.sell_orders.keys())) / 2
        
        # Calculate theoretical basket value
        theoretical_value = 0
        for component, quantity in basket_components.items():
            if component not in state.order_depths:
                return
                
            component_depth = state.order_depths[component]
            if not component_depth.buy_orders or not component_depth.sell_orders:
                return
                
            component_mid = (max(component_depth.buy_orders.keys()) + min(component_depth.sell_orders.keys())) / 2
            theoretical_value += component_mid * quantity
        
        # Calculate the spread
        spread = basket_mid - theoretical_value
        
        # Store spread history
        if not isinstance(state.traderData, dict):
            state.traderData = {}
        state.traderData['spread_history'] = []

        if 'spread_history' not in state.traderData:
            state.traderData['spread_history'] = []
        
        state.traderData['spread_history'].append(spread)
        
        # Calculate z-score if we have enough history
        if len(state.traderData['spread_history']) < 30:
            return
            
        mean_spread = sum(state.traderData['spread_history']) / len(state.traderData['spread_history'])
        std_spread = (sum((x - mean_spread) ** 2 for x in state.traderData['spread_history'][-25:]) / 25) ** 0.5
        
        if std_spread == 0:
            return
            
        z_score = (spread - mean_spread) / std_spread
        
        # Trading logic based on z-score
        target_position = 20
        if z_score > 2 and basket_position > -self.LIMIT[product]:
            # Spread is too high: Sell basket, buy components
            orders_to_place = min(target_position, self.LIMIT[product] + basket_position)
            if orders_to_place > 0:
                result.setdefault(product, []).append(Order(product, max(basket_depth.buy_orders.keys()), -orders_to_place))
                
                for component, quantity in basket_components.items():
                    component_depth = state.order_depths[component]
                    if component_depth.sell_orders:
                        result.setdefault(component, []).append(
                            Order(component, min(component_depth.sell_orders.keys()), quantity * orders_to_place)
                        )
        
        elif z_score < -2 and basket_position < self.LIMIT[product]:
            # Spread is too low: Buy basket, sell components
            orders_to_place = min(target_position, self.LIMIT[product] - basket_position)
            if orders_to_place > 0:
                result.setdefault(product, []).append(Order(product, min(basket_depth.sell_orders.keys()), orders_to_place))
                
                for component, quantity in basket_components.items():
                    component_depth = state.order_depths[component]
                    if component_depth.buy_orders:
                        result.setdefault(component, []).append(
                            Order(component, max(component_depth.buy_orders.keys()), -quantity * orders_to_place)
                        )

    
    def black_scholes_call(self, S, K, T, r, sigma):
        """Calculate Black-Scholes price for a call option."""
        if T <= 0:
            return max(0, S - K)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    def implied_volatility(self, price, S, K, T, r):
        """Calculate implied volatility using the Black-Scholes model."""
        if price <= 0 or T <= 0:
            return None
            
        # Define the objective function (difference between market and model price)
        def objective(sigma):
            return self.black_scholes_call(S, K, T, r, sigma) - price
        
        try:
            # Use brentq method to find the root (implied volatility)
            return brentq(objective, 0.001, 5.0)
        except:
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
    
    def trade_volcanic_vouchers(self, state):
        """Simple and direct implementation to ensure we trade volcanic vouchers."""
        logger.print("Starting simplified volcanic voucher trading")
        
        result = {}
        
        # Define all voucher symbols
        voucher_symbols = [
            Product.VOLCANIC_ROCK_VOUCHER_9500,
            Product.VOLCANIC_ROCK_VOUCHER_9750,
            Product.VOLCANIC_ROCK_VOUCHER_10000,
            Product.VOLCANIC_ROCK_VOUCHER_10250,
            Product.VOLCANIC_ROCK_VOUCHER_10500,
        ]
        
        # Get volcanic rock price if available
        volcanic_rock_price = None
        if "VOLCANIC_ROCK" in state.order_depths:
            vr_depth = state.order_depths["VOLCANIC_ROCK"]
            if vr_depth.buy_orders and vr_depth.sell_orders:
                best_bid = max(vr_depth.buy_orders.keys())
                best_ask = min(vr_depth.sell_orders.keys())
                volcanic_rock_price = (best_bid + best_ask) / 2
                logger.print(f"VOLCANIC_ROCK price: {volcanic_rock_price}")
        
        # Process each voucher
        for symbol in voucher_symbols:
            if symbol not in state.order_depths:
                logger.print(f"{symbol} not in order depths")
                continue
            
            depth = state.order_depths[symbol]
            if not depth.buy_orders and not depth.sell_orders:
                logger.print(f"{symbol} has no orders")
                continue
            
            # Get current position
            position = state.position.get(symbol, 0)
            logger.print(f"Current position for {symbol}: {position}")
            
            orders = []
            
            # Simple market making strategy - place orders at both sides
            if depth.buy_orders:
                best_bid = max(depth.buy_orders.keys())
                # Sell if we have room in our position
                if position > -200:
                    sell_quantity = min(depth.buy_orders[best_bid], 200 + position)
                    if sell_quantity > 0:
                        orders.append(Order(symbol, best_bid, -sell_quantity))
                        logger.print(f"Selling {sell_quantity} of {symbol} at {best_bid}")
            
            if depth.sell_orders:
                best_ask = min(depth.sell_orders.keys())
                # Buy if we have room in our position
                if position < 200:
                    buy_quantity = min(abs(depth.sell_orders[best_ask]), 200 - position)
                    if buy_quantity > 0:
                        orders.append(Order(symbol, best_ask, buy_quantity))
                        logger.print(f"Buying {buy_quantity} of {symbol} at {best_ask}")
            
            if orders:
                result[symbol] = orders
        
        return result
    
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
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        
        # Analyze base IV and update trader object
        traderObject = self.analyze_base_iv(state, traderObject) or traderObject
        
        result = {}
        conversions = 0
        
        # Add Round 1 products trading
        for product in [Product.RAINFOREST_RESIN, Product.KELP, Product.SQUID_INK]:
            if product in state.order_depths:
                position = state.position.get(product, 0)
                
                # Calculate fair value based on product type
                if product == Product.RAINFOREST_RESIN:
                    fair_value = self.params[product]["fair_value"]
                elif product == Product.KELP:
                    fair_value = self.KELP_fair_value(state.order_depths[product], traderObject)
                elif product == Product.SQUID_INK:
                    fair_value = self.SQUID_INK_fair_value(state.order_depths[product], traderObject)
                
                if fair_value is not None:
                    # Take orders
                    take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                        product,
                        state.order_depths[product],
                        fair_value,
                        self.params[product]["take_width"],
                        position,
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
                take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                    product,
                    state.order_depths[product],
                    fair_value,
                    self.params[product]["take_width"],
                    position,
                    self.params[product]["prevent_adverse"],
                    self.params[product]["adverse_volume"],
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
        # Apply basket arbitrage
        self.basket_arbitrage(state, Product.PICNIC_BASKET1, BASKET1_PRODS, result)
        self.basket_arbitrage(state, Product.PICNIC_BASKET2, BASKET2_PRODS, result)
        
        # Apply statistical arbitrage
        self.basket_stat_arb(state, Product.PICNIC_BASKET1, BASKET1_PRODS, result)
        self.basket_stat_arb(state, Product.PICNIC_BASKET2, BASKET2_PRODS, result)

        try:
            volcanic_orders = self.trade_volcanic_vouchers(state)
            for symbol, orders in volcanic_orders.items():
                result[symbol] = orders
        except Exception as e:
            logger.print(f"Error in volcanic voucher trading: {str(e)}")
        
        # Store trader data
        trader_data = ""
        logger.flush(state, result, conversions, trader_data)
        
        return result, conversions, trader_data