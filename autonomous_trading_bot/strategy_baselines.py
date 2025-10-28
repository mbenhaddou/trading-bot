from typing import Dict
import numpy as np
import logging
logger = logging.getLogger(__name__)

class BaselineStrategy:
    """Base class for baseline strategies"""

    def __init__(self, config: Dict):
        self.config = config

    def reset(self): pass
    def execute(self, env, portfolio_manager) -> float:
        """Execute strategy and return final return"""
        raise NotImplementedError


class BuyAndHoldBaseline(BaselineStrategy):
    """Buy equal amounts at start, hold until end - CACHE-AWARE"""

    def execute(self, env, portfolio_manager) -> float:
        initial_value = portfolio_manager.get_portfolio_value()

        # Buy equal amounts at START (current cache position)
        market_data = env.data_source.get_latest_data()

        if not market_data:
            logging.warning("BuyAndHold: No market data available")
            return 0.0

        cash_per_symbol = initial_value / len(env.symbols)

        from order_execution import SimulatedOrderExecution
        executor = SimulatedOrderExecution(self.config)

        for symbol in env.symbols:
            price = market_data.get(symbol, {}).get('close', 0)
            if price > 0:
                qty = int(cash_per_symbol / price)
                if qty > 0:
                    order = executor.place_buy_order(symbol, qty, market_price=price)
                    if order:
                        portfolio_manager.update_portfolio([order], market_data)

        # Hold (advance cache and update valuations)
        for step in range(env.max_steps - 1):
            # CRITICAL: Advance cache to next timestep
            if not env.data_source.advance_cache():
                logging.debug(f"BuyAndHold: Cache exhausted at step {step}")
                break

            # Get new market data from advanced position
            market_data = env.data_source.get_latest_data()

            if not market_data:
                break

        # Calculate final return with latest market data
        final_value = portfolio_manager.get_portfolio_value(market_data)
        total_return = (final_value - initial_value) / initial_value

        logging.info(f"BuyAndHold: {initial_value:.2f} -> {final_value:.2f} ({total_return:.2%})")
        return total_return


class EqualWeightBaseline(BaselineStrategy):
    """Rebalance to equal weights every 10 steps - CACHE-AWARE"""

    def execute(self, env, portfolio_manager) -> float:
        initial_value = portfolio_manager.get_portfolio_value()

        from order_execution import SimulatedOrderExecution
        executor = SimulatedOrderExecution(self.config)

        for step in range(env.max_steps):
            # Get current market data
            market_data = env.data_source.get_latest_data()

            if not market_data:
                logging.warning(f"EqualWeight: No market data at step {step}")
                break

            if step % 10 == 0:  # Rebalance every 10 steps
                total_value = portfolio_manager.get_portfolio_value(market_data)
                target_per_symbol = total_value / len(env.symbols)

                for symbol in env.symbols:
                    price = market_data.get(symbol, {}).get('close', 0)
                    if price <= 0:
                        continue

                    current_qty = portfolio_manager.get_current_holdings().get(symbol, 0)
                    current_value = current_qty * price

                    diff = target_per_symbol - current_value
                    qty_to_trade = int(abs(diff) / price)

                    if qty_to_trade >= 5:  # Minimum trade size
                        if diff > 0:  # Buy
                            order = executor.place_buy_order(symbol, qty_to_trade, market_price=price)
                        else:  # Sell
                            qty_to_trade = min(qty_to_trade, current_qty)
                            if qty_to_trade > 0:
                                order = executor.place_sell_order(symbol, qty_to_trade, market_price=price)
                            else:
                                order = None

                        if order:
                            portfolio_manager.update_portfolio([order], market_data)

            # CRITICAL: Advance cache to next timestep (unless last step)
            if step < env.max_steps - 1:
                if not env.data_source.advance_cache():
                    logging.debug(f"EqualWeight: Cache exhausted at step {step}")
                    break

        # Calculate final return
        final_market_data = env.data_source.get_latest_data()
        final_value = portfolio_manager.get_portfolio_value(final_market_data)
        total_return = (final_value - initial_value) / initial_value

        logging.info(f"EqualWeight: {initial_value:.2f} -> {final_value:.2f} ({total_return:.2%})")
        return total_return


class RandomBaseline(BaselineStrategy):
    """Random allocation changes - CACHE-AWARE"""

    def execute(self, env, portfolio_manager) -> float:
        initial_value = portfolio_manager.get_portfolio_value()

        from order_execution import SimulatedOrderExecution
        executor = SimulatedOrderExecution(self.config)

        for step in range(env.max_steps):
            # Get current market data
            market_data = env.data_source.get_latest_data()

            if not market_data:
                logging.warning(f"Random: No market data at step {step}")
                break

            if step % 10 == 0:  # Trade every 10 steps
                total_value = portfolio_manager.get_portfolio_value(market_data)

                # Random weights
                weights = np.random.dirichlet(np.ones(len(env.symbols)))

                for symbol, weight in zip(env.symbols, weights):
                    price = market_data.get(symbol, {}).get('close', 0)
                    if price <= 0:
                        continue

                    target_value = total_value * weight
                    current_qty = portfolio_manager.get_current_holdings().get(symbol, 0)
                    current_value = current_qty * price

                    diff = target_value - current_value
                    qty_to_trade = int(abs(diff) / price)

                    if qty_to_trade >= 5:  # Minimum trade size
                        if diff > 0:  # Buy
                            order = executor.place_buy_order(symbol, qty_to_trade, market_price=price)
                        else:  # Sell
                            qty_to_trade = min(qty_to_trade, current_qty)
                            if qty_to_trade > 0:
                                order = executor.place_sell_order(symbol, qty_to_trade, market_price=price)
                            else:
                                order = None

                        if order:
                            portfolio_manager.update_portfolio([order], market_data)

            # CRITICAL: Advance cache to next timestep (unless last step)
            if step < env.max_steps - 1:
                if not env.data_source.advance_cache():
                    logging.debug(f"Random: Cache exhausted at step {step}")
                    break

        # Calculate final return
        final_market_data = env.data_source.get_latest_data()
        final_value = portfolio_manager.get_portfolio_value(final_market_data)
        total_return = (final_value - initial_value) / initial_value

        logging.info(f"Random: {initial_value:.2f} -> {final_value:.2f} ({total_return:.2%})")
        return total_return