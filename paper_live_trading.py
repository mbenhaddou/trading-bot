"""
Alpaca Paper Trading - Live RL Agent Test
Adapted for your existing temporal RL system

This script:
1. Loads the best trained model
2. Fetches daily historical data for context
3. Gets current market data from Alpaca every 5 minutes
4. Makes trading decisions using your RL agent
5. Executes trades through Alpaca paper trading API

Requirements:
- Alpaca paper trading account
- ALPACA_API_KEY and ALPACA_API_SECRET environment variables
- Best trained model checkpoint

Usage:
    export ALPACA_API_KEY="your_key"
    export ALPACA_API_SECRET="your_secret"

    python alpaca_live_trading.py --model models/best_model.pth --max-iterations 100
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
load_dotenv()

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError:
    print("ERROR: Alpaca SDK not installed. Install with:")
    print("  pip install alpaca-py")
    sys.exit(1)

# Your existing system
from autonomous_trading_bot.config import load_config
from autonomous_trading_bot.temporal_rl_system import (
    TemporalRLTradingSystem,
    TradingMode,
    AggregatedFeatureExtractor
)
from autonomous_trading_bot.unified_data_provider import create_data_provider
from autonomous_trading_bot.portfolio_management import PortfolioManagement

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LiveTradingAgent:
    """
    Live trading agent that uses your trained temporal RL system
    """

    def __init__(self, config_path: str, model_checkpoint_path: str):
        """
        Initialize live trading agent

        Args:
            config_path: Path to config.json
            model_checkpoint_path: Path to best model checkpoint
        """
        # Load configuration
        self.config = load_config(config_path)

        # Override mode to live
        self.config['mode'] = 'live'

        # Alpaca credentials
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_API_SECRET')

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables\n"
                "Get them from: https://alpaca.markets/docs/trading/getting-started/"
            )

        # Initialize Alpaca clients
        logger.info("Connecting to Alpaca paper trading...")
        self.trading_client = TradingClient(
            self.api_key,
            self.secret_key,
            paper=True  # PAPER TRADING ONLY
        )
        self.data_client = StockHistoricalDataClient(
            self.api_key,
            self.secret_key
        )

        # Verify connection
        try:
            account = self.trading_client.get_account()
            logger.info(f"✓ Connected to Alpaca paper trading")
            logger.info(f"  Account: {account.account_number}")
            logger.info(f"  Status: {account.status}")
        except Exception as e:
            raise ValueError(f"Failed to connect to Alpaca: {e}")

        # Trading universe
        self.symbols = self.config.get('symbols_to_trade', [])
        logger.info(f"Trading universe: {self.symbols}")

        # Load trained RL system
        logger.info(f"Loading model from {model_checkpoint_path}")
        self.rl_system = self._load_model(model_checkpoint_path)

        # Feature extractor (matches training)
        self.feature_extractor = AggregatedFeatureExtractor()

        # Historical data cache (daily bars for context)
        self.historical_data = {}
        self.historical_lookback_days = 90  # 3 months of daily data

        # Decision interval (5 minutes)
        self.decision_interval_seconds = 5 * 60

        # Track last decision time
        self.last_decision_time = None

        # Performance tracking
        self.decisions = []
        self.trades = []
        self.initial_portfolio_value = None

        # Track returns for Sharpe calculation
        self._recent_returns = []
        self._peak_value = None
        self._last_portfolio_value = None

    def _load_model(self, checkpoint_path: str) -> TemporalRLTradingSystem:
        """Load trained RL model"""

        # Create RL system (in inference mode)
        rl_system = TemporalRLTradingSystem(self.config)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Load agent state
        rl_system.agent.strategy_policy.load_state_dict(checkpoint['strategy_policy'])
        rl_system.agent.allocation_policy.load_state_dict(checkpoint['allocation_policy'])
        rl_system.agent.execution_policy.load_state_dict(checkpoint['execution_policy'])

        # Set to inference mode (no exploration, no training)
        rl_system.agent.set_mode(TradingMode.INFERENCE)
        rl_system.agent.set_exploration_noise(0.0)  # No exploration in live trading

        logger.info(f"✓ Model loaded successfully")
        logger.info(f"  Training steps: {checkpoint.get('training_steps', 'unknown')}")
        logger.info(f"  Episodes trained: {checkpoint.get('episodes_trained', 'unknown')}")

        return rl_system

    def fetch_historical_data(self):
        """Fetch daily historical data for all symbols (using yfinance for free tier)"""
        logger.info("Fetching historical daily data...")

        # Use yfinance instead of Alpaca for historical data (works with free tier)
        try:
            import yfinance as yf

            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.historical_lookback_days)

            for symbol in self.symbols:
                try:
                    # Download from Yahoo Finance
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        interval='1d'
                    )

                    if df is not None and not df.empty:
                        # Normalize column names to lowercase
                        df.columns = [c.lower() for c in df.columns]

                        # Keep only OHLCV
                        df = df[['open', 'high', 'low', 'close', 'volume']].copy()

                        self.historical_data[symbol] = df
                        logger.info(f"  {symbol}: {len(df)} daily bars from Yahoo Finance")
                    else:
                        logger.warning(f"  No Yahoo Finance data for {symbol}")
                        self.historical_data[symbol] = self._create_synthetic_daily_data()

                except Exception as e:
                    logger.warning(f"  Error fetching {symbol} from Yahoo Finance: {e}")
                    self.historical_data[symbol] = self._create_synthetic_daily_data()

            logger.info("✓ Historical data loaded from Yahoo Finance")

        except ImportError:
            logger.warning("yfinance not installed, using synthetic data")
            logger.warning("Install with: pip install yfinance")
            for symbol in self.symbols:
                self.historical_data[symbol] = self._create_synthetic_daily_data()

        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            # Create synthetic data as fallback
            for symbol in self.symbols:
                self.historical_data[symbol] = self._create_synthetic_daily_data()

    def _create_synthetic_daily_data(self, days: int = 90) -> pd.DataFrame:
        """Create synthetic daily data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        base_price = 100.0

        returns = np.random.normal(0, 0.02, days)
        prices = base_price * np.exp(np.cumsum(returns))

        return pd.DataFrame({
            'open': prices * np.random.uniform(0.98, 1.02, days),
            'high': prices * np.random.uniform(1.01, 1.05, days),
            'low': prices * np.random.uniform(0.95, 0.99, days),
            'close': prices,
            'volume': np.random.uniform(1e6, 1e7, days)
        }, index=dates)

    def get_current_market_data(self) -> Dict:
        """
        Get current market data from Alpaca (using IEX feed for free tier)

        Returns:
            Dict with current prices and quotes
        """
        market_data = {}

        try:
            # Use latest trades instead of quotes (more reliable for free tier)
            for symbol in self.symbols:
                try:
                    # Get latest trade from IEX feed (free tier)
                    from alpaca.data.requests import StockLatestTradeRequest

                    trade_request = StockLatestTradeRequest(
                        symbol_or_symbols=symbol,
                        feed='iex'  # Use IEX feed for free tier
                    )

                    trades = self.data_client.get_stock_latest_trade(trade_request)

                    if symbol in trades:
                        trade = trades[symbol]
                        price = float(trade.price)

                        market_data[symbol] = {
                            'open': price,
                            'high': price,
                            'low': price,
                            'close': price,
                            'bid': price * 0.999,  # Estimate bid/ask
                            'ask': price * 1.001,
                            'volume': int(trade.size),
                            'last_trade': price,
                            'timestamp': trade.timestamp.isoformat() if hasattr(trade.timestamp, 'isoformat') else str(
                                trade.timestamp)
                        }
                    else:
                        logger.warning(f"No trade data for {symbol}")
                        market_data[symbol] = None

                except Exception as e:
                    logger.warning(f"Error getting data for {symbol}: {e}")
                    market_data[symbol] = None

        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            # Return None for all symbols
            for symbol in self.symbols:
                market_data[symbol] = None

        return market_data

    def get_account_info(self) -> Dict:
        """Get current account information"""
        account = self.trading_client.get_account()

        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value)
        }

    def get_current_positions(self) -> Dict:
        """Get current positions"""
        try:
            positions = self.trading_client.get_all_positions()

            position_dict = {}
            for position in positions:
                position_dict[position.symbol] = {
                    'qty': float(position.qty),
                    'avg_entry_price': float(position.avg_entry_price),
                    'market_value': float(position.market_value),
                    'cost_basis': float(position.cost_basis),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc)
                }

            return position_dict

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}

    def get_current_state(self, market_data: Dict) -> np.ndarray:
        """
        Get current state for RL agent (60-dimensional)

        Matches the state format from training
        """
        try:
            # Get account and positions
            account = self.get_account_info()
            positions = self.get_current_positions()

            current_value = account['portfolio_value']

            # Calculate daily return
            if self._last_portfolio_value:
                daily_return = (current_value - self._last_portfolio_value) / self._last_portfolio_value
            else:
                daily_return = 0
                self._last_portfolio_value = current_value

            # Track returns
            self._recent_returns.append(daily_return)
            if len(self._recent_returns) > 100:
                self._recent_returns.pop(0)

            # Estimate Sharpe
            if len(self._recent_returns) > 10:
                returns_array = np.array(self._recent_returns)
                sharpe = np.mean(returns_array) / (np.std(returns_array) + 1e-8)
            else:
                sharpe = 0

            # Calculate drawdown
            if self._peak_value is None:
                self._peak_value = current_value
            else:
                self._peak_value = max(self._peak_value, current_value)

            drawdown = (self._peak_value - current_value) / self._peak_value if self._peak_value > 0 else 0

            # Calculate position ratio
            position_values = [
                float(positions.get(sym, {}).get('market_value', 0))
                for sym in self.symbols
            ]
            total_position_value = sum(position_values)
            position_ratio = total_position_value / current_value if current_value > 0 else 0

            # Portfolio state
            portfolio_state = {
                'cash': account['cash'],
                'total_value': current_value,
                'unrealized_pnl': current_value - self.initial_portfolio_value,
                'positions': {sym: positions.get(sym, {}).get('qty', 0) for sym in self.symbols},
                'position_ratio': position_ratio,
                'daily_return': daily_return,
                'drawdown': drawdown,
                'sharpe_estimate': sharpe
            }

            # Extract features using your feature extractor
            state = self.feature_extractor.extract_features(
                self.historical_data,
                market_data,
                portfolio_state
            )

            # Ensure 60 dimensions
            if state.shape[0] != 60:
                logger.warning(f"State dimension mismatch: {state.shape[0]} != 60")
                state = np.pad(state, (0, max(0, 60 - state.shape[0])), 'constant')[:60]

            return state

        except Exception as e:
            logger.error(f"Error creating state: {e}")
            return np.zeros(60, dtype=np.float32)

    def make_decision(self, state: np.ndarray, step: int) -> Dict:
        """
        Use RL agent to make trading decision

        Args:
            state: Current state observation (60-dimensional)
            step: Current step (for temporal hierarchy)

        Returns:
            Decision dict with allocations and execution params
        """
        try:
            # Get action from agent (deterministic, no exploration)
            action_dict = self.rl_system.agent.select_action(
                state,
                step,
                self.symbols,
                deterministic=True
            )

            decision = {
                'timestamp': datetime.now(),
                'strategy': action_dict['strategy'],
                'allocation': action_dict['allocation'],
                'execution': action_dict['execution']
            }

            logger.info(f"Decision made:")
            logger.info(f"  Strategy: {decision['strategy'].regime} ({decision['strategy'].sentiment:.2f})")
            logger.info(
                f"  Top allocations: {sorted(decision['allocation'].items(), key=lambda x: x[1], reverse=True)[:3]}")

            self.decisions.append(decision)

            return decision

        except Exception as e:
            logger.error(f"Error making decision: {e}")
            # Return safe default (equal weight)
            return {
                'timestamp': datetime.now(),
                'allocation': {sym: 1.0 / len(self.symbols) for sym in self.symbols}
            }

    def execute_rebalance(self, decision: Dict, market_data: Dict):
        """
        Execute portfolio rebalance based on decision

        Args:
            decision: Decision from RL agent
            market_data: Current market data
        """
        account = self.get_account_info()
        positions = self.get_current_positions()

        target_allocation = decision['allocation']
        portfolio_value = account['portfolio_value']

        logger.info(f"Executing rebalance. Portfolio value: ${portfolio_value:.2f}")

        # Calculate target positions
        orders_to_place = []

        for symbol in self.symbols:
            target_weight = target_allocation.get(symbol, 0)

            # Skip CASH pseudo-symbol
            if symbol == 'CASH':
                continue

            target_value = portfolio_value * target_weight

            # Current position
            current_qty = positions.get(symbol, {}).get('qty', 0)

            # Get current price
            if not market_data.get(symbol):
                logger.warning(f"Skipping {symbol} - no market data")
                continue

            current_price = market_data[symbol].get('close', 0)

            if current_price <= 0:
                logger.warning(f"Skipping {symbol} - invalid price")
                continue

            current_value = current_qty * current_price
            value_diff = target_value - current_value

            # Calculate shares to trade
            shares_diff = int(value_diff / current_price)

            # Minimum trade threshold ($100)
            if abs(value_diff) < 100:
                continue

            # Create order
            if shares_diff > 0:
                # Buy
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=shares_diff,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                orders_to_place.append((symbol, order_data, 'BUY', shares_diff))

            elif shares_diff < 0:
                # Sell
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=abs(shares_diff),
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                orders_to_place.append((symbol, order_data, 'SELL', abs(shares_diff)))

        # Place orders
        for symbol, order_data, side, qty in orders_to_place:
            try:
                order = self.trading_client.submit_order(order_data)
                logger.info(f"✓ {side} {qty} {symbol} (Order ID: {order.id})")

                self.trades.append({
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'side': side,
                    'qty': qty,
                    'order_id': order.id,
                    'status': order.status
                })

            except Exception as e:
                logger.error(f"✗ Failed to place {side} order for {symbol}: {e}")

    def run_live_trading_loop(self, max_iterations: Optional[int] = None):
        """
        Main live trading loop

        Args:
            max_iterations: Maximum iterations (None for infinite)
        """
        logger.info("=" * 80)
        logger.info("STARTING LIVE TRADING")
        logger.info("=" * 80)

        # Fetch initial historical data
        self.fetch_historical_data()

        # Get initial account state
        account = self.get_account_info()
        self.initial_portfolio_value = account['portfolio_value']
        self._last_portfolio_value = self.initial_portfolio_value

        logger.info(f"Initial portfolio value: ${self.initial_portfolio_value:.2f}")
        logger.info(f"Initial cash: ${account['cash']:.2f}")
        logger.info(
            f"Decision interval: {self.decision_interval_seconds}s ({self.decision_interval_seconds / 60:.0f} minutes)")

        iteration = 0
        step = 0  # For temporal hierarchy

        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1

                logger.info(f"\n{'=' * 80}")
                logger.info(f"Iteration {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'=' * 80}")

                # Check if market is open
                clock = self.trading_client.get_clock()
                if not clock.is_open:
                    next_open = clock.next_open
                    logger.info(f"Market is closed. Next open: {next_open}")
                    time.sleep(60)
                    continue

                # Get current market data
                market_data = self.get_current_market_data()

                # Get current state
                state = self.get_current_state(market_data)

                # Make decision
                decision = self.make_decision(state, step)

                # Execute trades
                self.execute_rebalance(decision, market_data)

                # Log performance
                account = self.get_account_info()
                current_value = account['portfolio_value']
                total_return = (current_value - self.initial_portfolio_value) / self.initial_portfolio_value

                logger.info(f"\nPerformance:")
                logger.info(f"  Portfolio value: ${current_value:.2f}")
                logger.info(f"  Total return: {total_return:.2%}")
                logger.info(f"  Cash: ${account['cash']:.2f}")
                logger.info(f"  Decisions made: {len(self.decisions)}")
                logger.info(f"  Trades executed: {len(self.trades)}")

                # Update step counter
                step += 1

                # Update historical data periodically (every 12 iterations = 1 hour)
                if iteration % 12 == 0:
                    logger.info("\nRefreshing historical data...")
                    self.fetch_historical_data()

                # Wait for next decision interval
                logger.info(f"\nWaiting {self.decision_interval_seconds}s until next decision...")
                time.sleep(self.decision_interval_seconds)

        except KeyboardInterrupt:
            logger.info("\n" + "=" * 80)
            logger.info("STOPPING LIVE TRADING (Ctrl+C)")
            logger.info("=" * 80)
            self.print_summary()

        except Exception as e:
            logger.error(f"\nError in trading loop: {e}", exc_info=True)
            self.print_summary()

    def print_summary(self):
        """Print trading session summary"""
        logger.info("\n" + "=" * 80)
        logger.info("TRADING SESSION SUMMARY")
        logger.info("=" * 80)

        # Final account state
        account = self.get_account_info()
        final_value = account['portfolio_value']
        total_return = (final_value - self.initial_portfolio_value) / self.initial_portfolio_value

        logger.info(f"\nPerformance:")
        logger.info(f"  Initial value: ${self.initial_portfolio_value:.2f}")
        logger.info(f"  Final value: ${final_value:.2f}")
        logger.info(f"  Total return: {total_return:.2%}")
        logger.info(f"  P&L: ${final_value - self.initial_portfolio_value:+.2f}")

        # Trades summary
        logger.info(f"\nActivity:")
        logger.info(f"  Total decisions: {len(self.decisions)}")
        logger.info(f"  Total trades: {len(self.trades)}")

        if self.trades:
            logger.info(f"\nRecent trades:")
            for trade in self.trades[-10:]:
                logger.info(f"  {trade['timestamp'].strftime('%H:%M:%S')} - "
                            f"{trade['side']:4} {trade['qty']:3} {trade['symbol']}")

        logger.info("=" * 80)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Live RL Trading on Alpaca Paper Trading',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with best model for 100 iterations
    python alpaca_live_trading.py --model models/best_model.pth --max-iterations 100

    # Run indefinitely (Ctrl+C to stop)
    python alpaca_live_trading.py --model models/best_model.pth

    # Use specific config file
    python alpaca_live_trading.py --config my_config.json --model models/best_model.pth

Environment variables required:
    ALPACA_API_KEY - Your Alpaca API key
    ALPACA_API_SECRET - Your Alpaca API secret
        """
    )

    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to config file (default: config.json)')
    parser.add_argument('--max-iterations', type=int, default=None,
                        help='Maximum trading iterations (default: infinite)')
    parser.add_argument('--interval', type=int, default=5,
                        help='Decision interval in minutes (default: 5)')

    args = parser.parse_args()

    # Check for API keys
    if not os.getenv('ALPACA_API_KEY') or not os.getenv('ALPACA_API_SECRET'):
        print("\n❌ ERROR: Alpaca API credentials not found!")
        print("\nPlease set environment variables:")
        print("  export ALPACA_API_KEY='your_key_here'")
        print("  export ALPACA_API_SECRET='your_secret_here'")
        print("\nGet your keys from: https://alpaca.markets/docs/trading/")
        sys.exit(1)

    # Create and run agent
    try:
        model="training_logs/checkpoints/best_model.pth"
        agent = LiveTradingAgent(args.config, model)
        agent.decision_interval_seconds = args.interval * 60
        agent.run_live_trading_loop(max_iterations=args.max_iterations)
    except Exception as e:
        logger.error(f"Failed to start live trading: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()