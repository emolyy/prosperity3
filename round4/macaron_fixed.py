from datamodel import Order, TradingState, ConversionObservation
import json

class Logger:
    def __init__(self):
        self.logs = ""
    
    def print(self, *objects, sep=" ", end="\n"):
        self.logs += sep.join([str(o) for o in objects]) + end
    
    def flush(self, state=""):
        print(json.dumps({"logs": self.logs, "state": state}))
        self.logs = ""

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
        from round4 import logger

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
            logger.flush(state, {SYMBOL: orders}, conversions_used, trader_data)
            return result

        cost = compute_conversion_cost(obs)
        revenue = compute_conversion_revenue(obs)
        
        logger.print(f"Position: {pos}, Cost: {cost}, Revenue: {revenue}")

        order_depth = state.order_depths.get(SYMBOL)
        if not order_depth:
            logger.print(f"No order depth for {SYMBOL}")
            result = {SYMBOL: orders}, conversions_used, trader_data
            logger.flush(state, {SYMBOL: orders}, conversions_used, trader_data)
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
                logger.print(f"Selling {qty} at {best_bid}")

        if best_ask is not None and revenue - best_ask >= SPREAD_THRESHOLD:
            qty = min(CONVERSION_LIMIT, pos + POSITION_LIMIT)
            if qty > 0:
                orders.append(Order(SYMBOL, best_ask, qty))
                conversions_used += qty
                pos -= qty
                logger.print(f"Buying {qty} at {best_ask}")

        # This must be the last line before returning
        result = {SYMBOL: orders}, conversions_used, trader_data
        logger.flush(state, {SYMBOL: orders}, conversions_used, trader_data)
        return result
