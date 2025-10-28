"""
CORRECTED portfolio_management.py - Fixes circular import and market data handling
"""
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

# Removed circular import: from risk_manager import RiskManager
#from autonomous_trading_bot.performance_tracker import PerformanceTracker
import logging
logger = logging.getLogger(__name__)

class PortfolioManagement:
    def __init__(self, config, risk_manager=None):
        """Initialize with optional risk_manager to avoid circular dependency"""
        self.config = config
        self.risk_manager = risk_manager  # Optional dependency injection
        self.initial_capital = config.get('initial_capital', 10000.0)
        self.current_cash = self.initial_capital
        self.holdings = {}
        self.portfolio_value = self.initial_capital
        self.max_drawdown_percent = config.get('max_drawdown_percent', 10.0)
        self.daily_loss_limit_percent = config.get('daily_loss_limit_percent', 5.0)
        self.peak_portfolio_value = self.initial_capital
        self.daily_start_value = self.initial_capital

        # Track last known prices for emergency fallback
        self.last_market_data = {}
        self.last_known_prices = {}
        self.realized_pnl = 0.0
        self.snapshots = []

        #self.performance_tracker = PerformanceTracker()

        logging.info(f"PortfolioManagement initialized with capital: ${self.initial_capital:,.2f}")

    def reset(self):
        """Reset portfolio to initial state for new episode/trading period."""
        self.current_cash = self.initial_capital
        self.holdings = {}
        self.portfolio_value = self.initial_capital
        self.peak_portfolio_value = self.initial_capital
        self.daily_start_value = self.initial_capital

        # Clear cached market data
        self.last_market_data = {}
        self.last_known_prices = {}

        # Reset performance tracker if it has a reset method


        logging.info(f"Portfolio reset to initial capital: ${self.initial_capital:,.2f}")

    def update_portfolio(self, executed_orders, current_market_data=None):
        """Updates portfolio based on executed orders and current market data."""

        # Cache market data for future calculations
        if current_market_data:
            self.last_market_data = current_market_data
            # Update last known prices
            for symbol, data in current_market_data.items():
                if isinstance(data, dict) and 'close' in data:
                    self.last_known_prices[symbol] = data['close']

        for order in executed_orders:
            symbol = order.symbol
            quantity = float(order.qty)
            price = float(order.filled_avg_price) if order.filled_avg_price else 0.0

            if order.side == 'buy':
                self.current_cash -= quantity * price
                self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
                logging.info(f"Bought {quantity} of {symbol} at ${price:.2f}. Cash: ${self.current_cash:.2f}")
            elif order.side == 'sell':
                    avg = float(self.last_known_prices.get(order.symbol, getattr(order, 'filled_avg_price', 0.0)))
                    if avg > 0:
                        self.realized_pnl += (float(order.filled_avg_price) - avg) * float(order.filled_qty)
            self._calculate_portfolio_value(current_market_data)
            self._check_risk_limits()
            self._snapshot(current_market_data)

    def _snapshot(self, market_data=None):
        rec = {
            'timestamp': datetime.now().isoformat(),
            'cash': self.current_cash,
            'holdings_value': self.get_holdings_value(market_data or self.last_market_data),
            'total_value': self.portfolio_value,
            'realized_pnl': self.realized_pnl
        }
        self.snapshots.append(rec)

    def export_snapshots(self, path: str | Path = "portfolio_snapshots.csv"):
        if not self.snapshots: return
        df = pd.DataFrame(self.snapshots)
        df.to_csv(path, index=False)

    def _calculate_portfolio_value(self, current_market_data=None):
        """Calculates the current total value of the portfolio with robust handling."""

        # Use provided market data or fall back to cached data
        market_data = current_market_data if current_market_data else self.last_market_data

        assets_value = 0

        if market_data and self.holdings:
            for symbol, qty in self.holdings.items():
                price = 0

                # Try to get price from market data
                if symbol in market_data:
                    if isinstance(market_data[symbol], dict):
                        price = market_data[symbol].get('close', 0)
                    else:
                        price = float(market_data[symbol]) if market_data[symbol] else 0

                # Fallback to last known price if current price unavailable
                if price <= 0 and symbol in self.last_known_prices:
                    price = self.last_known_prices[symbol]
                    logging.warning(f"Using last known price for {symbol}: ${price:.2f}")

                # Emergency fallback if still no price
                if price <= 0:
                    price = self.get_emergency_liquidation_price(symbol)
                    logging.warning(f"Using emergency price for {symbol}: ${price:.2f}")

                assets_value += qty * price

        self.portfolio_value = self.current_cash + assets_value
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)

        logging.debug(
            f"Portfolio value: cash=${self.current_cash:.2f}, "
            f"assets=${assets_value:.2f}, total=${self.portfolio_value:.2f}"
        )

    def _check_risk_limits(self):
        """Checks if the portfolio has hit any predefined risk limits."""
        current_drawdown = ((self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value) * 100

        if current_drawdown > self.max_drawdown_percent:
            # Only log once every 100 violations
            if not hasattr(self, '_drawdown_violations'):
                self._drawdown_violations = 0
            self._drawdown_violations += 1

            if self._drawdown_violations % 100 == 1:
                logging.warning(f"Drawdown: {current_drawdown:.2f}% exceeds {self.max_drawdown_percent:.1f}% "
                                f"(violation #{self._drawdown_violations})")

        daily_loss = ((self.daily_start_value - self.portfolio_value) / self.daily_start_value) * 100

        if daily_loss > self.daily_loss_limit_percent:
            if not hasattr(self, '_daily_loss_violations'):
                self._daily_loss_violations = 0
            self._daily_loss_violations += 1

            if self._daily_loss_violations % 100 == 1:
                logging.warning(f"Daily loss: {daily_loss:.2f}% exceeds {self.daily_loss_limit_percent:.1f}% "
                                f"(violation #{self._daily_loss_violations})")
    def rebalance_portfolio(self, current_market_data, target_allocations=None):
        """Rebalances the portfolio based on target allocations."""
        if not current_market_data:
            logging.warning("Cannot rebalance: no current market data available.")
            return []

        if target_allocations is None:
            # Equal-weight rebalancing for current holdings
            if not self.holdings:
                logging.info("No holdings to rebalance.")
                return []
            num_assets = len(self.holdings)
            if num_assets == 0:
                return []
            target_per_asset = (self.portfolio_value - self.current_cash) / num_assets
            target_allocations = {symbol: target_per_asset for symbol in self.holdings}

        rebalance_orders = []
        for symbol, target_value in target_allocations.items():
            current_qty = self.holdings.get(symbol, 0)

            # Get current price
            current_price = 0
            if symbol in current_market_data:
                if isinstance(current_market_data[symbol], dict):
                    current_price = current_market_data[symbol].get('close', 0)
                else:
                    current_price = float(current_market_data[symbol]) if current_market_data[symbol] else 0

            if current_price <= 0:
                logging.warning(f"Cannot rebalance {symbol}: no valid price available.")
                continue

            current_value = current_qty * current_price
            difference = target_value - current_value

            threshold = self.config.get('rebalance_threshold_value', 100)
            if abs(difference) > threshold:
                quantity_to_trade = int(abs(difference) / current_price)
                if quantity_to_trade == 0:
                    continue

                if difference > 0:  # Need to buy
                    rebalance_orders.append({
                        'action': 'buy',
                        'symbol': symbol,
                        'quantity': quantity_to_trade,
                        'price': current_price
                    })
                    logging.info(f"Rebalance BUY signal for {symbol}: {quantity_to_trade} shares")
                else:  # Need to sell
                    quantity_to_trade = min(quantity_to_trade, current_qty)
                    if quantity_to_trade > 0:
                        rebalance_orders.append({
                            'action': 'sell',
                            'symbol': symbol,
                            'quantity': quantity_to_trade,
                            'price': current_price
                        })
                        logging.info(f"Rebalance SELL signal for {symbol}: {quantity_to_trade} shares")

        return rebalance_orders

    def get_current_holdings(self):
        """Returns current stock holdings."""
        return self.holdings.copy()

    def get_portfolio_value(self, current_market_data=None):
        """Returns the total portfolio value with proper None handling."""

        # Update with new market data if provided
        if current_market_data:
            self.last_market_data = current_market_data
            # Update last known prices
            for symbol, data in current_market_data.items():
                if isinstance(data, dict) and 'close' in data:
                    self.last_known_prices[symbol] = data['close']

        # Use provided data, cached data, or emergency fallback
        market_data = current_market_data or self.last_market_data or {}

        # Calculate holdings value
        holdings_value = 0.0

        if self.holdings and market_data:
            for symbol, qty in self.holdings.items():
                price = 0

                if symbol in market_data:
                    if isinstance(market_data[symbol], dict):
                        price = market_data[symbol].get('close', 0)
                    else:
                        price = float(market_data[symbol]) if market_data[symbol] else 0

                # Fallback to last known or emergency price
                if price <= 0:
                    if symbol in self.last_known_prices:
                        price = self.last_known_prices[symbol]
                    else:
                        price = self.get_emergency_liquidation_price(symbol)
                        logging.warning(f"No market data for {symbol}, using fallback price ${price:.2f}")

                holdings_value += qty * price

        self.portfolio_value = self.current_cash + holdings_value
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)

        return self.portfolio_value

    def get_holdings_value(self, current_market_data: Dict = None) -> float:
        """Get the current market value of all holdings."""
        market_data = current_market_data if current_market_data else self.last_market_data

        if not market_data or not self.holdings:
            return 0.0

        holdings_value = 0.0
        for symbol, qty in self.holdings.items():
            price = 0

            if symbol in market_data:
                if isinstance(market_data[symbol], dict):
                    price = market_data[symbol].get('close', 0)
                else:
                    price = float(market_data[symbol]) if market_data[symbol] else 0

            if price <= 0 and symbol in self.last_known_prices:
                price = self.last_known_prices[symbol]

            holdings_value += qty * price

        return holdings_value

    def get_emergency_liquidation_price(self, symbol):
        """Get fallback price for emergency liquidation when market data unavailable."""
        # Use last known price if available
        if symbol in self.last_known_prices:
            return self.last_known_prices[symbol]

        # Estimate based on initial capital and holdings
        total_holdings_count = len(self.holdings)
        if total_holdings_count > 0:
            estimated_price = self.initial_capital / (total_holdings_count * 100)
            return max(estimated_price, 1.0)  # At least $1

        return 50.0  # Default fallback price

    def get_available_cash(self):
        """Returns the available cash for trading."""
        return self.current_cash

    def get_risk_per_trade_amount(self):
        """Calculates the maximum amount to risk per trade based on config."""
        risk_percent = self.config.get('risk_per_trade_percent', 1.0) / 100
        return self.portfolio_value * risk_percent

    def validate_and_size_position(self, signal, market_data=None):
        """Validate trade against risk limits using injected risk manager."""
        if self.risk_manager:
            return self.risk_manager.validate_trade(signal, self, market_data)
        else:
            # Basic validation if no risk manager
            if signal.get('quantity', 0) > 0:
                # Check if we have enough cash for buy orders
                if signal.get('action') == 'buy':
                    required_cash = signal['quantity'] * signal.get('price', 0)
                    if required_cash > self.current_cash:
                        signal['quantity'] = int(self.current_cash / signal['price'])
                return signal if signal['quantity'] > 0 else None
            return None

    def place_initial_orders_via_broker(self, order_execution, current_market_data,
                                        symbols=None, fraction=1.0):
        """Place initial buy orders via the provided OrderExecution instance."""
        symbols = symbols or self.config.get('symbols_to_trade', [])
        if not symbols:
            logging.warning("No symbols available for initial seeding.")
            return []

        usable_cash = max(0.0, float(self.current_cash)) * float(fraction)
        if usable_cash <= 0:
            logging.info("No cash available for initial seeding.")
            return []

        # Build price list
        symbol_prices = []
        for s in symbols:
            price_data = current_market_data.get(s, {})
            if isinstance(price_data, dict):
                price = price_data.get('close', 0)
            else:
                price = float(price_data) if price_data else 0

            if price and price > 0:
                symbol_prices.append((s, float(price)))

        if not symbol_prices:
            logging.warning("No valid market prices available to seed holdings.")
            return []

        per_symbol_cash = usable_cash / len(symbol_prices)
        placed_orders = []

        for symbol, price in symbol_prices:
            qty = int(per_symbol_cash / price)
            if qty <= 0:
                logging.info(f"Not enough cash to buy 1 share of {symbol} (price ${price:.2f})")
                continue

            try:
                order = order_execution.place_buy_order(symbol, qty, price=None)
                if order:
                    # Track the order
                    if not hasattr(order_execution, 'executed_orders'):
                        order_execution.executed_orders = []

                    if not any(getattr(o, 'id', None) == getattr(order, 'id', None)
                               for o in order_execution.executed_orders):
                        order_execution.executed_orders.append(order)

                    placed_orders.append(order)
                    logging.info(f"Placed initial buy order for {qty} {symbol}")
                else:
                    logging.error(f"Failed to place initial buy for {symbol}")
            except Exception as e:
                logging.error(f"Error placing initial buy for {symbol}: {e}")

        return placed_orders

    def reset_daily_values(self):
        """Reset daily tracking values (call at market open)"""
        self.daily_start_value = self.portfolio_value
        logging.info(f"Daily values reset. Starting value: ${self.daily_start_value:.2f}")