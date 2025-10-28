import alpaca_trade_api as tradeapi
import threading
from types import SimpleNamespace
import uuid
import time
import numpy as np
from datetime import datetime
from collections import deque

import logging
logger = logging.getLogger(__name__)


class OrderExecution:
    def __init__(self, config):
        self.config = config
        self.api = tradeapi.REST(
            config["api_key"],
            config["api_secret"],
            config["base_url"],
            api_version='v2'
        )
        self.executed_orders = []
        self._lock = threading.Lock()
        logging.info("OrderExecution module initialized.")

    def place_order_with_retry(self, symbol, quantity, side, order_type='limit', limit_price=None, max_retries=3):
        for attempt in range(max_retries):
            try:
                lp = limit_price
                if order_type == 'limit' and lp is not None:
                    if side == 'buy':
                        lp = lp * 1.001
                    else:
                        lp = lp * 0.999

                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    type=order_type,
                    time_in_force='gtc',
                    limit_price=lp if order_type == 'limit' else None
                )

                # wait briefly for fill
                for _ in range(10):
                    order = self.api.get_order(order.id)
                    if getattr(order, 'status', None) == 'filled':
                        with self._lock:
                            self.executed_orders.append(order)
                        logging.info(f"Order filled and recorded: {side} {quantity} {symbol} (id={order.id})")
                        return order
                    time.sleep(1)

                # cancel if not filled
                if getattr(order, 'status', None) not in ['filled', 'cancelled']:
                    try:
                        self.api.cancel_order(order.id)
                        logging.warning(f"Order not filled and cancelled: id={order.id}")
                    except Exception as e:
                        logging.error(f"Failed to cancel order {order.id}: {e}")

            except Exception as e:
                logging.error(f"Order attempt {attempt + 1} failed for {side} {symbol}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        logging.error(f"Failed to place order after {max_retries} attempts: {side} {symbol}")
        return None

    def place_buy_order(self, symbol, quantity, price=None):
        return self.place_order_with_retry(symbol, quantity, 'buy',
                                           'limit' if price else 'market', price)

    def place_sell_order(self, symbol, quantity, price=None):
        """Places a sell order."""
        try:
            if price:
                # Limit order if price is specified
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',
                    type='limit',
                    time_in_force='gtc',
                    limit_price=price
                )
            else:
                # Market order if no price specified
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )

            # Wait for fill confirmation
            for _ in range(10):
                order = self.api.get_order(order.id)
                if getattr(order, 'status', None) == 'filled':
                    with self._lock:
                        self.executed_orders.append(order)
                    logging.info(f"Placed sell order and recorded (filled): {quantity} {symbol}, id={order.id}")
                    return order
                time.sleep(1)

            # If not filled in time, try to cancel
            if getattr(order, 'status', None) not in ['filled', 'cancelled']:
                try:
                    self.api.cancel_order(order.id)
                    logging.warning(f"Sell order not filled in time and cancelled: id={order.id}")
                except Exception as e:
                    logging.error(f"Failed to cancel sell order {order.id}: {e}")

            return None

        except Exception as e:
            logging.error(f"Error placing sell order for {symbol}: {e}")
            return None

    def get_executed_orders(self):
        """Return and clear executed orders (atomically)."""
        with self._lock:
            orders = list(self.executed_orders)
            self.executed_orders.clear()
        return orders

    def cancel_order(self, order_id):
        """Cancels an open order."""
        try:
            self.api.cancel_order(order_id)
            logging.info(f"Cancelled order {order_id}")
            with self._lock:
                self.executed_orders = [order for order in self.executed_orders if order.id != order_id]
            return True
        except Exception as e:
            logging.error(f"Error cancelling order {order_id}: {e}")
            return False


class SimulatedOrderExecution:
    """Enhanced simulation executor with realistic market dynamics"""

    def __init__(self, config):
        self.config = config
        self.executed_orders = []
        self.pending_orders = deque()
        self.order_history = []
        self.order_id_counter = 0

        # Simulation parameters
        self.commission_per_share = config.get('simulation', {}).get('commission_per_share', 0.005)
        self.slippage_bps = config.get('simulation', {}).get('slippage_bps', 5)
        self.use_realistic_fills = config.get('simulation', {}).get('use_realistic_fills', True)
        self.market_impact_model = config.get('simulation', {}).get('market_impact_model', 'linear')

        # Track order statistics
        self.total_orders_placed = 0
        self.total_orders_filled = 0
        self.total_orders_rejected = 0
        self.total_partial_fills = 0

        logging.info("SimulatedOrderExecution initialized (enhanced simulation mode)")

    def _generate_order_id(self):
        """Generate unique order ID"""
        self.order_id_counter += 1
        return f"SIM_{uuid.uuid4().hex[:8]}_{self.order_id_counter}"

    def _calculate_market_impact(self, symbol, quantity, current_price, side):
        """Calculate market impact on price"""
        if self.market_impact_model == 'none':
            return 0

        # Simple linear impact model
        # Impact increases with order size
        base_impact = 0.0001  # 1 basis point base impact
        size_factor = np.log1p(quantity) / 10  # Logarithmic scaling

        if self.market_impact_model == 'square_root':
            # Square root model (common in practice)
            impact = base_impact * np.sqrt(quantity / 1000)
        else:
            # Linear model
            impact = base_impact * size_factor

        # Buy orders push price up, sell orders push price down
        if side == 'buy':
            return current_price * impact
        else:
            return -current_price * impact

    def _simulate_order_latency(self):
        """Simulate network and processing latency"""
        # Random latency between 1-100ms
        latency_ms = np.random.uniform(1, 100)
        time.sleep(latency_ms / 1000)

    def _make_order_obj(self, symbol, qty, side, filled_price, status='filled',
                        filled_qty=None, commission=0):
        """Create a realistic order object"""
        order_id = self._generate_order_id()
        filled_qty = filled_qty or qty

        order = SimpleNamespace(
            id=order_id,
            symbol=symbol,
            qty=float(qty),
            filled_qty=float(filled_qty),
            side=side,
            filled_avg_price=float(filled_price),
            status=status,
            timestamp=datetime.now(),
            commission=float(commission),
            created_at=time.time(),
            filled_at=time.time() if status == 'filled' else None,
            order_type='market',
            time_in_force='gtc'
        )

        # Store in history
        self.order_history.append({
            'id': order_id,
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'filled_qty': filled_qty,
            'price': filled_price,
            'status': status,
            'commission': commission,
            'timestamp': datetime.now()
        })

        return order

    def _apply_realistic_slippage(self, price, side, volatility=0.02):
        """Apply realistic slippage based on market conditions"""
        if not self.use_realistic_fills:
            return price

        # Base slippage
        slippage_factor = self.slippage_bps / 10000

        # Add random component based on volatility
        random_factor = np.random.normal(0, volatility * slippage_factor)

        # Adverse selection: buy orders tend to execute at ask (higher)
        # sell orders tend to execute at bid (lower)
        if side == 'buy':
            # Add spread cost (typically 0.01% to 0.05% for liquid stocks)
            spread_cost = np.random.uniform(0.0001, 0.0005)
            return price * (1 + abs(random_factor) + spread_cost)
        else:
            spread_cost = np.random.uniform(0.0001, 0.0005)
            return price * (1 - abs(random_factor) - spread_cost)

    def _simulate_partial_fill(self, requested_qty, market_volume):
        """Simulate partial order fills based on market liquidity"""
        if not self.use_realistic_fills:
            return requested_qty

        # Probability of partial fill increases with order size relative to volume
        if market_volume > 0:
            size_ratio = requested_qty / market_volume

            # If order is more than 1% of daily volume, chance of partial fill
            if size_ratio > 0.01:
                partial_fill_prob = min(0.5, size_ratio * 10)

                if np.random.random() < partial_fill_prob:
                    # Fill between 50% and 90% of order
                    fill_percentage = np.random.uniform(0.5, 0.9)
                    filled_qty = int(requested_qty * fill_percentage)
                    self.total_partial_fills += 1
                    logging.info(f"[SIM] Partial fill: {filled_qty}/{requested_qty} shares")
                    return filled_qty

        return requested_qty

    def _should_reject_order(self, symbol, quantity, side):
        """Determine if order should be rejected"""
        if not self.use_realistic_fills:
            return False

        # Base rejection probability
        reject_prob = 0.01

        # Increase rejection probability for large orders
        if quantity > 10000:
            reject_prob += 0.02

        # Random rejection
        if np.random.random() < reject_prob:
            self.total_orders_rejected += 1
            logging.warning(f"[SIM] Order rejected: {side} {quantity} {symbol}")
            return True

        return False

    def place_buy_order(self, symbol, quantity, price=None, market_price=None):
        """Simulate buy order with realistic execution"""
        self.total_orders_placed += 1

        # Check for order rejection
        if self._should_reject_order(symbol, quantity, 'buy'):
            return None

        # Simulate latency
        if self.use_realistic_fills:
            self._simulate_order_latency()

        # Determine execution price
        base_price = market_price if market_price is not None else (price if price is not None else 0.0)

        if base_price <= 0:
            logging.error(f"[SIM] Invalid price for {symbol}: {base_price}")
            return None

        # Calculate volatility (simplified - in practice, use historical data)
        volatility = np.random.uniform(0.01, 0.04)  # 1-4% volatility

        # Apply slippage
        execution_price = self._apply_realistic_slippage(base_price, 'buy', volatility)

        # Apply market impact
        market_impact = self._calculate_market_impact(symbol, quantity, execution_price, 'buy')
        execution_price += market_impact

        # Simulate partial fills (assume average daily volume)
        assumed_volume = np.random.uniform(1e6, 10e6)  # 1M to 10M shares
        filled_quantity = self._simulate_partial_fill(quantity, assumed_volume)

        # Calculate commission
        commission = filled_quantity * self.commission_per_share

        # Create order object
        order = self._make_order_obj(
            symbol, quantity, 'buy', execution_price,
            status='filled' if filled_quantity == quantity else 'partial',
            filled_qty=filled_quantity,
            commission=commission
        )

        self.executed_orders.append(order)
        self.total_orders_filled += 1

        logging.info(f"[SIM] BUY executed: {filled_quantity}/{quantity} {symbol} @ ${execution_price:.2f} "
                     f"(impact: ${market_impact:.4f}, commission: ${commission:.2f})")

        return order

    def place_sell_order(self, symbol, quantity, price=None, market_price=None):
        """Simulate sell order with realistic execution"""
        self.total_orders_placed += 1

        # Check for order rejection
        if self._should_reject_order(symbol, quantity, 'sell'):
            return None

        # Simulate latency
        if self.use_realistic_fills:
            self._simulate_order_latency()

        # Determine execution price
        base_price = market_price if market_price is not None else (price if price is not None else 0.0)

        if base_price <= 0:
            logging.error(f"[SIM] Invalid price for {symbol}: {base_price}")
            return None

        # Calculate volatility
        volatility = np.random.uniform(0.01, 0.04)

        # Apply slippage
        execution_price = self._apply_realistic_slippage(base_price, 'sell', volatility)

        # Apply market impact
        market_impact = self._calculate_market_impact(symbol, quantity, execution_price, 'sell')
        execution_price += market_impact

        # Simulate partial fills
        assumed_volume = np.random.uniform(1e6, 10e6)
        filled_quantity = self._simulate_partial_fill(quantity, assumed_volume)

        # Calculate commission
        commission = filled_quantity * self.commission_per_share

        # Create order object
        order = self._make_order_obj(
            symbol, quantity, 'sell', execution_price,
            status='filled' if filled_quantity == quantity else 'partial',
            filled_qty=filled_quantity,
            commission=commission
        )

        self.executed_orders.append(order)
        self.total_orders_filled += 1

        logging.info(f"[SIM] SELL executed: {filled_quantity}/{quantity} {symbol} @ ${execution_price:.2f} "
                     f"(impact: ${market_impact:.4f}, commission: ${commission:.2f})")

        return order

    def place_limit_order(self, symbol, quantity, limit_price, side='buy'):
        """Simulate limit order (may not fill immediately)"""
        self.total_orders_placed += 1

        # Store as pending order
        pending_order = {
            'symbol': symbol,
            'quantity': quantity,
            'limit_price': limit_price,
            'side': side,
            'timestamp': datetime.now(),
            'id': self._generate_order_id()
        }

        self.pending_orders.append(pending_order)
        logging.info(f"[SIM] Limit order placed: {side} {quantity} {symbol} @ ${limit_price:.2f}")

        # Return order object with pending status
        return self._make_order_obj(symbol, quantity, side, limit_price, status='pending')

    def get_executed_orders(self):
        """Return and clear executed orders"""
        orders = list(self.executed_orders)
        self.executed_orders.clear()
        return orders

    def get_order_statistics(self):
        """Get simulation order statistics"""
        return {
            'total_placed': self.total_orders_placed,
            'total_filled': self.total_orders_filled,
            'total_rejected': self.total_orders_rejected,
            'total_partial_fills': self.total_partial_fills,
            'fill_rate': self.total_orders_filled / max(1, self.total_orders_placed),
            'rejection_rate': self.total_orders_rejected / max(1, self.total_orders_placed),
            'partial_fill_rate': self.total_partial_fills / max(1, self.total_orders_filled)
        }

    def export_order_history(self, filepath=None):
        """Export order history to CSV"""
        import pandas as pd
        import os

        if not self.order_history:
            logging.warning("No order history to export")
            return

        df = pd.DataFrame(self.order_history)

        if filepath is None:
            filepath = os.path.join(os.path.dirname(__file__),
                                    f'order_history_{datetime.now():%Y%m%d_%H%M%S}.csv')

        df.to_csv(filepath, index=False)
        logging.info(f"Order history exported to {filepath}")