from datamodel import Order, TradingState, ConversionObservation

POSITION_LIMIT = 75
CONVERSION_LIMIT = 10
STORAGE_COST = 0.1  # per unit per timestamp
PRODUCT = "MAGNIFICENT_MACARONS"
SYMBOL = "MAGNIFICENT_MACARONS"
SPREAD_THRESHOLD = 1.0  # derived from robust stats across all days

def compute_conversion_cost(obs: ConversionObservation) -> float:
    return obs.askPrice + obs.transportFees + obs.importTariff

def compute_conversion_revenue(obs: ConversionObservation) -> float:
    return obs.bidPrice - obs.transportFees - obs.exportTariff

class Trader:
    def __init__(self):
        self.prev_position = 0  # used to account for holding costs if desired

    def run(self, state: TradingState):
        orders = []
        conversions_used = 0
        pos = state.position.get(PRODUCT, 0)
        obs = state.observations.conversionObservations.get(PRODUCT)

        if not obs:
            return {SYMBOL: orders}, conversions_used

        cost = compute_conversion_cost(obs)
        revenue = compute_conversion_revenue(obs)

        order_depth = state.order_depths.get(SYMBOL)
        if not order_depth:
            return {SYMBOL: orders}, conversions_used

        best_bid = max(order_depth.buy_orders.keys(), default=None)
        best_ask = min(order_depth.sell_orders.keys(), default=None)

        # SELL to market if cost to buy via conversion is cheap
        if best_bid is not None and best_bid - cost >= SPREAD_THRESHOLD:
            qty = min(CONVERSION_LIMIT, POSITION_LIMIT - pos)
            if qty > 0:
                orders.append(Order(SYMBOL, best_bid, -qty))
                conversions_used += qty
                pos += qty  # update simulated position

        # BUY from market if conversion revenue is high
        if best_ask is not None and revenue - best_ask >= SPREAD_THRESHOLD:
            qty = min(CONVERSION_LIMIT, pos + POSITION_LIMIT)
            if qty > 0:
                orders.append(Order(SYMBOL, best_ask, qty))
                conversions_used += qty
                pos -= qty  # update simulated position

        return {SYMBOL: orders}, conversions_used
