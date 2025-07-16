from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, ConversionObservation
from typing import List, Any, Dict
import json

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

POSITION_LIMIT = 75
CONVERSION_LIMIT = 10
STORAGE_COST = 0.1  
PRODUCT = "MAGNIFICENT_MACARONS"
SYMBOL = "MAGNIFICENT_MACARONS"
SPREAD_THRESHOLD = 1.0  # derived from robust stats across all days

def compute_conversion_cost(obs: ConversionObservation) -> float:
    return obs.askPrice + obs.transportFees + obs.importTariff

def compute_conversion_revenue(obs: ConversionObservation) -> float:
    return obs.bidPrice - obs.transportFees - obs.exportTariff

class Trader:
    def __init__(self):
        self.prev_position = 0  # holding costs if desired
        self.logger = Logger()

    def run(self, state: TradingState):
        # Initialize empty lists and variables
        orders = []
        conversions_used = 0
        pos = state.position.get(PRODUCT, 0)
        obs = state.observations.conversionObservations.get(PRODUCT)
        
        # Store trader data for returning later
        trader_data = state.traderData
        
        if not obs:
            logger.print(f"No observations for {PRODUCT}")
            result = {SYMBOL: orders}, conversions_used, trader_data
            result = {SYMBOL: orders}
            logger.flush(state, result, conversions_used, trader_data)
            return result

        cost = compute_conversion_cost(obs)
        revenue = compute_conversion_revenue(obs)
        
        logger.print(f"Position: {pos}, Cost: {cost}, Revenue: {revenue}")

        order_depth = state.order_depths.get(SYMBOL)
        if not order_depth:
            logger.print(f"No order depth for {SYMBOL}")
            result = {SYMBOL: orders}, conversions_used, trader_data
            result = {SYMBOL: orders}
            logger.flush(state, result, conversions_used, trader_data)
            return result

        best_bid = max(order_depth.buy_orders.keys(), default=None)
        best_ask = min(order_depth.sell_orders.keys(), default=None)
        
        logger.print(f"Best bid: {best_bid}, Best ask: {best_ask}")

        if best_bid is not None and best_bid - cost >= SPREAD_THRESHOLD:
            qty = min(CONVERSION_LIMIT, POSITION_LIMIT - pos)
            if qty > 0:
                orders.append(Order(SYMBOL, best_bid, -qty))
                conversions_used += qty
                pos += qty
                self.logger.print(f"Selling {qty} at {best_bid}")

        if best_ask is not None and revenue - best_ask >= SPREAD_THRESHOLD:
            qty = min(CONVERSION_LIMIT, pos + POSITION_LIMIT)
            if qty > 0:
                orders.append(Order(SYMBOL, best_ask, qty))
                conversions_used += qty
                pos -= qty
                logger.print(f"Buying {qty} at {best_ask}")

        # This must be the last line before returning
        result = {SYMBOL: orders}
        logger.flush(state, result, conversions_used, trader_data)

        return result
