"""IBKR-style dollar costs per order."""

from alpha_factory.contracts.sandbox import CostSpec


def order_cost(costs: CostSpec, shares: int, price: float, *, is_sell: bool) -> float:
    """Commission (per share with a minimum and a cap), half-spread and sell fees, in dollars.

    cost = min(max(q * 0.0035, 0.35), 1% of notional) + notional * half_spread_bps / 1e4
           + (notional * sell_fee_bps / 1e4 if the order is a sell)

    Short sells count as sells. The half-spread is charged here rather than moved into the fill
    price, so it is counted once.
    """
    notional = shares * price
    commission = min(
        max(shares * costs.commission_per_share, costs.min_commission_per_order),
        costs.max_commission_pct * notional,
    )
    cost = commission + notional * costs.half_spread_bps / 1e4
    if is_sell:
        cost += notional * costs.sell_fee_bps / 1e4
    return cost
