"""
Enhanced Temporal Abstraction RL Trading System
With explicit Training and Inference modes
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque, defaultdict
import logging
from autonomous_trading_bot.logging_setup import setup_logging
import gym
from gym import spaces
from enum import Enum
import logging as logger
from autonomous_trading_bot.order_execution import SimulatedOrderExecution
from autonomous_trading_bot.portfolio_management import PortfolioManagement
from autonomous_trading_bot.risk_manager import RiskManager
from autonomous_trading_bot.strategy_baselines import BuyAndHoldBaseline, EqualWeightBaseline, RandomBaseline

setup_logging(level="ERROR")

class TradingMode(Enum):
    """Trading system modes"""
    TRAINING = "training"
    EVALUATION = "evaluation"
    INFERENCE = "inference"
    BACKTEST = "backtest"


@dataclass
class StrategyDecision:
    """High-level strategy decision"""
    regime: str  # 'trending', 'ranging', 'volatile'
    sentiment: float  # -1 (bearish) to 1 (bullish)
    risk_mode: str  # 'aggressive', 'moderate', 'conservative'


@dataclass
class AllocationDecision:
    """Mid-level allocation decision"""
    weights: Dict[str, float]
    diversification: float


@dataclass
class ExecutionDecision:
    """Low-level execution decision"""
    urgency: float
    aggressiveness: float
    risk_reduction: float


class AggregatedFeatureExtractor:
    """Symbol-count agnostic feature extraction using aggregation"""

    def __init__(self):
        self.feature_history = deque(maxlen=1000)
        self.stats = {
            'mean': None,
            'std': None,
            'initialized': False
        }

    def extract_features(self,
                         historical_data: Dict[str, pd.DataFrame],
                         current_market_data: Dict,
                         portfolio_state: Dict) -> np.ndarray:
        """Extract fixed-size features regardless of symbol count"""

        # Collect per-symbol features
        symbol_features = []

        for symbol in sorted(historical_data.keys()):
            df = historical_data[symbol]
            if df.empty or len(df) < 2:
                symbol_features.append([0.0] * 10)
                continue

            try:
                close_prices = df['close'].values.ravel()
                features = self._extract_symbol_features(close_prices, df)

                if len(features) != 10:
                    logging.warning(f"Symbol {symbol} returned {len(features)} features, expected 10")
                    features = features[:10] + [0.0] * (10 - len(features))

                symbol_features.append(features)

            except Exception as e:
                logging.error(f"Error extracting features for {symbol}: {e}")
                symbol_features.append([0.0] * 10)

        if not symbol_features:
            return np.zeros(60, dtype=np.float32)

        try:
            symbol_features = np.array(symbol_features, dtype=np.float32)
        except ValueError as e:
            logging.error(f"Failed to create feature array: {e}")
            return np.zeros(60, dtype=np.float32)

        if len(symbol_features.shape) != 2 or symbol_features.shape[1] != 10:
            logging.error(f"Unexpected feature array shape: {symbol_features.shape}")
            return np.zeros(60, dtype=np.float32)

        # Aggregate across symbols
        aggregated = []

        for i in range(10):
            feature_values = symbol_features[:, i]
            aggregated.extend([
                float(np.mean(feature_values)),
                float(np.max(feature_values)),
                float(np.min(feature_values)),
                float(np.std(feature_values)),
                float(np.median(feature_values))
            ])

        # Add portfolio features (10 features)
        portfolio_features = self._extract_portfolio_features(
            portfolio_state, current_market_data
        )
        aggregated.extend(portfolio_features)

        features = np.array(aggregated, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = self._normalize(features)

        return features

    def _extract_symbol_features(self, close_prices: np.ndarray,
                                 df: pd.DataFrame) -> List[float]:
        """Extract exactly 10 features for a single symbol"""

        features = [0.0] * 10

        try:
            if close_prices is None or len(close_prices) < 2:
                return features

            close_prices = np.asarray(close_prices).flatten()

            # Feature 0: 1-bar return
            if len(close_prices) >= 2:
                features[0] = float(close_prices[-1] / close_prices[-2] - 1)

            # Feature 1: 5-bar return
            if len(close_prices) >= 6:
                features[1] = float(close_prices[-1] / close_prices[-6] - 1)

            # Feature 2: 20-bar return
            if len(close_prices) >= 21:
                features[2] = float(close_prices[-1] / close_prices[-21] - 1)

            # Feature 3: Volatility (std of returns)
            if len(close_prices) > 1:
                returns = np.diff(close_prices) / close_prices[:-1]
                returns = returns[~np.isnan(returns)]
                features[3] = float(np.std(returns)) if len(returns) > 0 else 0.0

            # Feature 4: Volume ratio
            if 'volume' in df.columns and len(df) > 1:
                current_vol = float(df['volume'].iloc[-1])
                avg_vol = float(df['volume'].iloc[:-1].mean())
                features[4] = float(current_vol / max(avg_vol, 1.0))

            # Feature 5: RSI
            rsi_value = self._calculate_rsi(close_prices)
            features[5] = float(rsi_value) if np.isfinite(rsi_value) else 0.5

            # Feature 6: SMA crossover
            if len(close_prices) >= 50:
                sma_20 = float(np.mean(close_prices[-20:]))
                sma_50 = float(np.mean(close_prices[-50:]))
                features[6] = float((sma_20 - sma_50) / sma_50) if sma_50 != 0 else 0.0

            # Feature 7: 10-bar momentum
            if len(close_prices) >= 10:
                features[7] = float((close_prices[-1] - close_prices[-10]) / close_prices[-10]) if close_prices[
                                                                                                       -10] != 0 else 0.0

            # Feature 8: Bollinger band position
            bb_position = self._bollinger_position(close_prices)
            features[8] = float(bb_position) if np.isfinite(bb_position) else 0.5

            # Feature 9: ATR
            if 'high' in df.columns and 'low' in df.columns:
                atr_value = self._calculate_atr(df)
                features[9] = float(atr_value) if np.isfinite(atr_value) else features[3]
            else:
                features[9] = features[3]

        except Exception as e:
            logging.warning(f"Error in feature extraction: {e}")
            # Return zero features on error
            return [0.0] * 10

        # Ensure all features are finite floats
        cleaned_features = []
        for i, f in enumerate(features[:10]):
            try:
                # Convert to float and check if finite
                val = float(f)
                if not np.isfinite(val):
                    val = 0.0
                cleaned_features.append(val)
            except (TypeError, ValueError):
                cleaned_features.append(0.0)

        # Ensure exactly 10 features
        while len(cleaned_features) < 10:
            cleaned_features.append(0.0)

        return cleaned_features[:10]

    def _extract_portfolio_features(self, portfolio_state: Dict,
                                    market_data: Dict) -> List[float]:
        """Extract portfolio-level features - returns list of 10 scalar floats"""

        try:
            cash = float(portfolio_state.get('cash', 0))
            total_value = float(portfolio_state.get('total_value', 1))
            positions = portfolio_state.get('positions', {})

            if positions and total_value > 0:
                position_values = [
                    float(qty) * float(market_data.get(sym, {}).get('close', 0))
                    for sym, qty in positions.items()
                ]
                max_position = float(max(position_values)) if position_values else 0.0
                concentration = max_position / total_value
            else:
                concentration = 0.0

            if positions:
                position_values = [
                    float(qty) * float(market_data.get(sym, {}).get('close', 0))
                    for sym, qty in positions.items()
                ]
                sum_values = sum(position_values)
                if sum_values > 0:
                    weights = np.array([v / sum_values for v in position_values])
                    hhi = float(np.sum(weights ** 2))
                    diversification = 1.0 - hhi
                else:
                    diversification = 0.0
            else:
                diversification = 0.0

            features = [
                cash / 100000,  # Normalized cash
                total_value / 100000,  # Normalized portfolio value
                portfolio_state.get('unrealized_pnl', 0) / 10000,  # Normalized PnL
                len(positions) / 20,  # Number of positions (normalized for max 20)
                portfolio_state.get('position_ratio', 0),  # Position/cash ratio
                concentration,
                diversification,
                portfolio_state.get('daily_return', 0) * 100,  # Daily return %
                portfolio_state.get('drawdown', 0),
                portfolio_state.get('sharpe_estimate', 0) / 3  # Normalized Sharpe
            ]

            # Ensure all features are finite scalar floats
            validated = []
            for f in features:
                try:
                    val = float(f)
                    if not np.isfinite(val):
                        val = 0.0
                    validated.append(val)
                except (TypeError, ValueError):
                    validated.append(0.0)

            # Ensure exactly 10 features
            while len(validated) < 10:
                validated.append(0.0)

            return validated[:10]

        except Exception as e:
            logging.error(f"Error in portfolio feature extraction: {e}")
            return [0.0] * 10

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI - returns scalar float"""
        try:
            if prices is None or len(prices) < period + 1:
                return 0.5

            prices = np.asarray(prices).flatten()
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = float(np.mean(gains[-period:]))
            avg_loss = float(np.mean(losses[-period:]))

            if avg_loss == 0:
                return 1.0

            rs = avg_gain / avg_loss
            rsi = 1 - (1 / (1 + rs))
            return float(rsi)
        except Exception as e:
            logging.debug(f"RSI calculation error: {e}")
            return 0.5

    def _bollinger_position(self, prices: np.ndarray, period: int = 20) -> float:
        """Position within Bollinger Bands - returns scalar float"""
        try:
            if prices is None or len(prices) < period:
                return 0.5

            prices = np.asarray(prices).flatten()
            recent = prices[-period:]
            mean = float(np.mean(recent))
            std = float(np.std(recent))

            if std == 0:
                return 0.5

            current = float(prices[-1])
            upper = mean + 2 * std
            lower = mean - 2 * std

            if upper == lower:
                position = 0.5
            else:
                position = (current - lower) / (upper - lower)

            return float(np.clip(position, 0, 1))
        except Exception as e:
            logging.debug(f"Bollinger position error: {e}")
            return 0.5

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR - returns scalar float"""
        try:
            if df is None or len(df) < period:
                return 0.0

            # CRITICAL: Use .ravel() to ensure 1D arrays regardless of pandas version
            high = df['high'].values.ravel().astype(float)
            low = df['low'].values.ravel().astype(float)
            close = df['close'].values.ravel().astype(float)

            # Verify shapes are 1D
            if high.ndim != 1 or low.ndim != 1 or close.ndim != 1:
                logging.warning(
                    f"ATR: Non-1D arrays detected - high:{high.shape}, low:{low.shape}, close:{close.shape}")
                high = high.flatten()
                low = low.flatten()
                close = close.flatten()

            # Create previous close array (shift by 1)
            prev_close = np.concatenate(([np.nan], close[:-1]))

            # Calculate True Range components
            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)

            # True Range is the maximum of the three
            tr = np.maximum(tr1, np.maximum(tr2, tr3))

            # Average True Range over period
            atr = float(np.nanmean(tr[-period:]))

            # Normalize by last close price
            last_close = float(close[-1])
            if last_close == 0 or not np.isfinite(last_close):
                return 0.0

            return float(atr / last_close)
        except Exception as e:
            logging.debug(f"ATR calculation error: {e}")
            return 0.0

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features with warm-up period"""

        self.feature_history.append(features)

        if len(self.feature_history) < 200:
            return np.clip(features, -10, 10)

        if not self.stats['initialized']:
            history = np.array(list(self.feature_history))
            self.stats['mean'] = np.mean(history, axis=0)
            self.stats['std'] = np.std(history, axis=0)
            self.stats['initialized'] = True

        alpha = 0.01
        self.stats['mean'] = alpha * features + (1 - alpha) * self.stats['mean']
        delta = np.abs(features - self.stats['mean'])
        self.stats['std'] = alpha * delta + (1 - alpha) * self.stats['std']

        normalized = (features - self.stats['mean']) / (self.stats['std'] + 1e-8)
        return np.clip(normalized, -5, 5)

class TemporalTradingEnvironment(gym.Env):
    """Trading environment for temporal abstraction agent - FIXED CACHE USAGE"""

    def __init__(self, config: Dict, data_source, portfolio_manager):
        super().__init__()

        self.config = config
        self.data_source = data_source
        self.portfolio_manager = portfolio_manager
        self.symbols = config.get('symbols_to_trade', [])

        self.feature_extractor = AggregatedFeatureExtractor()
        self.risk_manager = RiskManager(config)

        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(60,), dtype=np.float32
        )

        self.current_step = 0
        self.max_steps = config.get('max_episode_steps', 1000)
        self.last_portfolio_value = None

        self.episode_returns = []
        self.episode_sharpe = deque(maxlen=100)

    def reset(self, mode: str = 'train'):
        """
        Reset environment with explicit mode

        Args:
            mode: 'train', 'validation', or 'test'

        Returns:
            Initial state (60-dimensional)
        """
        logging.info(f"\n{'=' * 60}")
        logging.info(f"ENVIRONMENT RESET - {mode.upper()} MODE")
        logging.info(f"{'=' * 60}")

        self.current_step = 0
        self.episode_data = []
        self.episode_returns = []

        # Reset portfolio
        self.portfolio_manager.reset()

        # Initialize cache for this mode
        cache_initialized = self.data_source.initialize_cache_for_episode(
            mode=mode,
            lookback_bars=100
        )

        if not cache_initialized:
            logging.error("Failed to initialize data cache")
            return np.zeros(60, dtype=np.float32)

        # Verify cache mode
        if not self.data_source.cache_mode:
            logging.error("Cache mode not active after initialization")
            return np.zeros(60, dtype=np.float32)

        # Optional: Validate no look-ahead (disable in production)
        if self.config.get('validate_no_lookahead', False):
            self.data_source.validate_no_lookahead()

        # Get initial state
        state = self._get_state()
        self.last_portfolio_value = self.portfolio_manager.get_portfolio_value()

#        progress = self.data_source.get_progress()
        logging.info(f"Episode initialized:")
        logging.info(f"  Mode: {mode}")
#        logging.info(f"  Start: {progress['current_timestamp']}")
#        logging.info(f"  Length: {progress['bars_remaining']} bars")
        logging.info(f"{'=' * 60}\n")

        return state

    def step(self, action_dict):
        """Execute hierarchical action - FIXED cache advancement"""

        # Get market data from cache at CURRENT position
        market_data = self.data_source.get_latest_data()

        if not market_data:
            logging.warning("No market data available at current cache position")
            return self._get_state(), 0, True, {'reason': 'no_market_data'}

        # Execute trades based on current market data
        trades = self._execute_allocation(
            action_dict['allocation'],
            action_dict['execution'],
            market_data
        )

        # Calculate rewards
        strategy_reward = self._calculate_strategy_reward(action_dict['strategy'])
        allocation_reward = self._calculate_allocation_reward(trades)
        execution_reward = self._calculate_execution_reward(trades)

        # CRITICAL: Advance to next timestep in cache AFTER executing trades
        cache_advanced = self.data_source.advance_cache()

        if not cache_advanced:
            logging.info(f"Reached end of cached data at step {self.current_step}")
            return self._get_state(), execution_reward, True, {'reason': 'cache_exhausted'}

        # Get next state from NEW cache position
        next_state = self._get_state()

        # Check if done
        self.current_step += 1
        done = self._check_done()

        info = {
            'trades': trades,
            'strategy_reward': strategy_reward,
            'allocation_reward': allocation_reward,
            'execution_reward': execution_reward,
            'cache_index': self.data_source.current_index
        }

        reward = execution_reward

        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """Get current state ensuring no future information"""

        # Get historical data (only past)
        historical_data = {}
        symbols = self.config.get('symbols_to_trade', [])
        interval = self.config.get('simulation', {}).get('timeframe', '1Min')
        for symbol in symbols:
            hist = self.data_source.get_historical_data(symbol, interval, 100)
            if hist is not None and not hist.empty:
                historical_data[symbol] = hist

        # Get current market data
        current_data = self.data_source.get_latest_data()

        if not current_data:
            logging.warning("No current market data available for state")
            return np.zeros(60, dtype=np.float32)

        # Recalculate portfolio value with current market prices
        current_value = self.portfolio_manager.get_portfolio_value(current_data)

        # Calculate daily return
        if hasattr(self, 'last_portfolio_value') and self.last_portfolio_value:
            daily_return = (current_value - self.last_portfolio_value) / self.last_portfolio_value
        else:
            daily_return = 0
            self.last_portfolio_value = current_value

        # Estimate Sharpe
        if not hasattr(self, 'episode_returns'):
            self.episode_returns = []

        self.episode_returns.append(daily_return)

        if len(self.episode_returns) > 10:
            sharpe = np.mean(self.episode_returns) / (np.std(self.episode_returns) + 1e-8)
        else:
            sharpe = 0

        # Portfolio state
        portfolio_state = {
            'cash': self.portfolio_manager.get_available_cash(),
            'total_value': current_value,
            'unrealized_pnl': current_value - self.portfolio_manager.initial_capital,
            'positions': self.portfolio_manager.get_current_holdings(),
            'position_ratio': self._calculate_position_ratio(),
            'daily_return': daily_return,
            'drawdown': self._calculate_drawdown(),
            'sharpe_estimate': sharpe
        }

        # Extract features
        state = self.feature_extractor.extract_features(
            historical_data, current_data, portfolio_state
        )

        return state

    def _execute_allocation(self, weights: Dict[str, float],
                            execution_params: Dict,
                            market_data: Dict) -> List[Dict]:
        """Execute allocation with execution parameters"""

        trades = []
        portfolio_value = self.portfolio_manager.get_portfolio_value(market_data)
        current_holdings = self.portfolio_manager.get_current_holdings()

        # Calculate target positions
        for symbol, target_weight in weights.items():
            if symbol == 'CASH':
                continue

            if symbol not in market_data:
                continue

            current_price = market_data[symbol].get('close', 0)
            if current_price <= 0:
                continue

            # Target value for this symbol
            target_value = portfolio_value * target_weight
            current_qty = current_holdings.get(symbol, 0)
            current_value = current_qty * current_price

            # Calculate trade needed
            value_diff = target_value - current_value
            qty_to_trade = int(abs(value_diff) / current_price)

            MIN_TRADE_SIZE = 5

            if qty_to_trade < MIN_TRADE_SIZE:
                continue

            # Create signal
            signal = {
                'symbol': symbol,
                'action': 'buy' if value_diff > 0 else 'sell',
                'quantity': qty_to_trade,
                'price': current_price,
                'urgency': execution_params['urgency']
            }

            # Validate with risk manager
            validated = self.risk_manager.validate_trade(
                signal, self.portfolio_manager, market_data
            )

            if validated:
                # ✅ FIXED: Pass market_data to _execute_trade
                trade = self._execute_trade(validated, execution_params, market_data)
                if trade:
                    trades.append(trade)

        return trades

    def _execute_trade(self, signal: Dict, execution_params: Dict,
                       market_data: Dict) -> Optional[Dict]:
        """Execute single trade with market data for portfolio updates"""

        sim_executor = SimulatedOrderExecution(self.config)

        # Adjust price based on urgency/aggressiveness
        price = signal['price']
        if execution_params['aggressiveness'] > 0.7:
            price *= (1.001 if signal['action'] == 'buy' else 0.999)

        if signal['action'] == 'buy':
            order = sim_executor.place_buy_order(
                signal['symbol'], signal['quantity'], market_price=price
            )
        else:
            order = sim_executor.place_sell_order(
                signal['symbol'], signal['quantity'], market_price=price
            )

        if order:
            # ✅ FIXED: Pass market_data to update_portfolio
            self.portfolio_manager.update_portfolio([order], market_data)
            return {
                'symbol': signal['symbol'],
                'side': signal['action'],
                'qty': order.filled_qty,
                'price': order.filled_avg_price,
                'execution_quality': self._assess_execution(order, signal)
            }

        return None

    def _assess_execution(self, order, signal) -> float:
        """Execution quality metric"""
        expected_price = signal.get('price', 0.0)
        actual_price = getattr(order, 'filled_avg_price', expected_price)

        if expected_price <= 0:
            return 0.0

        slippage = abs(actual_price - expected_price) / expected_price
        quality = float(np.exp(-slippage * 200.0))
        return float(np.clip(quality, 0.0, 1.0))

    def _calculate_strategy_reward(self, strategy: StrategyDecision) -> float:
        """Long-horizon reward for strategy decisions"""

        if len(self.episode_returns) >= 60:
            hour_returns = self.episode_returns[-60:]
            hour_return = np.sum(hour_returns)
            hour_sharpe = np.mean(hour_returns) / (np.std(hour_returns) + 1e-8)
            reward = hour_sharpe * 10
        else:
            reward = 0

        return reward

    def _calculate_allocation_reward(self, trades: List[Dict]) -> float:
        """Mid-horizon reward for allocation decisions"""

        if len(self.episode_returns) >= 10:
            recent_return = np.sum(self.episode_returns[-10:])
        else:
            recent_return = 0

        # Reward concentration
        holdings = self.portfolio_manager.get_current_holdings()
        if holdings:
            market_data = self.data_source.get_latest_data()
            total_value = self.portfolio_manager.get_portfolio_value(market_data)

            weights = []
            for symbol, qty in holdings.items():
                price = market_data.get(symbol, {}).get('close', 0)
                if price > 0:
                    weights.append((qty * price) / total_value)

            if weights:
                entropy = -sum(w * np.log(w + 1e-10) for w in weights)
                concentration_bonus = (2.0 - entropy) * 0.5
            else:
                concentration_bonus = -1.0
        else:
            concentration_bonus = -1.0

        reward = recent_return * 100 + concentration_bonus
        return reward

    def _calculate_execution_reward(self, trades: List[Dict]) -> float:
        """Reward relative to baseline performance"""

        current_value = self.portfolio_manager.get_portfolio_value()
        if self.last_portfolio_value:
            portfolio_return = (current_value - self.last_portfolio_value) / self.last_portfolio_value
        else:
            portfolio_return = 0

        self.last_portfolio_value = current_value

        if hasattr(self, 'baseline_return_benchmark'):
            relative_performance = portfolio_return - self.baseline_return_benchmark
            baseline_bonus = relative_performance * 50
        else:
            baseline_bonus = 0

        if trades:
            avg_execution_quality = np.mean([t['execution_quality'] for t in trades])
            execution_bonus = (avg_execution_quality - 0.8)
        else:
            execution_bonus = -0.01

        cost_penalty = -len(trades) * 0.0001

        reward = portfolio_return * 100 + baseline_bonus + execution_bonus + cost_penalty

        return np.clip(reward, -2, 2)

    def _calculate_position_ratio(self) -> float:
        """Calculate position/cash ratio"""
        holdings = self.portfolio_manager.get_current_holdings()
        cash = self.portfolio_manager.get_available_cash()
        total = self.portfolio_manager.get_portfolio_value()

        if total <= 0:
            return 0

        position_value = total - cash
        return position_value / total

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown"""
        current = self.portfolio_manager.get_portfolio_value()
        peak = getattr(self.portfolio_manager, 'peak_portfolio_value', current)

        return (peak - current) / peak if peak > 0 else 0

    def _check_done(self) -> bool:
        """Check if episode should end"""

        # Time limit
        if self.current_step >= self.max_steps:
            return True

        # Drawdown limit
        if self._calculate_drawdown() > 0.25:
            return True

        # Portfolio wipeout
        if self.portfolio_manager.get_portfolio_value() < self.portfolio_manager.initial_capital * 0.5:
            return True

        return False

class TemporalAbstractionAgent:
    """Enhanced three-level hierarchical agent with explicit modes"""

    def __init__(self, num_symbols: int = 20, lr: float = 3e-4):
        self.num_symbols = num_symbols
        self.mode = TradingMode.TRAINING  # Default mode
        self.exploration_noise = 0.3  # Default exploration noise

        # Networks
        self.strategy_policy = PolicyNetwork(input_dim=60, output_dim=5)
        self.allocation_policy = PolicyNetwork(input_dim=65, output_dim=num_symbols + 1)
        execution_input_dim = 60 + 5 + (num_symbols + 1)
        self.execution_policy = PolicyNetwork(input_dim=execution_input_dim, output_dim=3)

        # Optimizers
        self.strategy_optimizer = optim.Adam(self.strategy_policy.parameters(), lr=lr)
        self.allocation_optimizer = optim.Adam(self.allocation_policy.parameters(), lr=lr)
        self.execution_optimizer = optim.Adam(self.execution_policy.parameters(), lr=lr)

        # Memory buffers with max sizes
        self.strategy_memory = deque(maxlen=5000)
        self.allocation_memory = deque(maxlen=10000)
        self.execution_memory = deque(maxlen=20000)

        # Current decisions - FIXED: Initialize with correct dimensions
        self.current_strategy = np.zeros(5, dtype=np.float32)  # 5 elements
        self.current_allocation = np.zeros(num_symbols + 1, dtype=np.float32)  # num_symbols + 1 elements

        # Value estimates
        self.strategy_value = 0
        self.allocation_value = 0
        self.execution_value = 0

        # Training parameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2

        # Metadata storage
        self._last_strategy_info = None
        self._last_allocation_info = None
        self._last_execution_info = None

        # Training statistics
        self.training_steps = 0
        self.episodes_trained = 0

        logging.info(f"Agent initialized in {self.mode.value} mode")

    def set_mode(self, mode: TradingMode):
        """Set the agent's operating mode"""
        self.mode = mode

        if mode == TradingMode.TRAINING:
            self._set_training_mode()
        elif mode in [TradingMode.EVALUATION, TradingMode.INFERENCE, TradingMode.BACKTEST]:
            self._set_eval_mode()

        logging.info(f"Agent mode set to: {mode.value}")

    def set_exploration_noise(self, noise: float):
        """
        Set exploration noise for stochastic policy

        Controls randomness during action selection:
        - High noise (0.3+) = more exploration (different actions)
        - Low noise (0.01) = more exploitation (consistent actions)

        Args:
            noise: Exploration noise level (0.0 to 1.0)
        """
        self.exploration_noise = max(0.0, min(1.0, noise))
        logging.debug(f"Exploration noise: {self.exploration_noise:.3f}")

    def _set_training_mode(self):
        """Set all networks to training mode"""
        self.strategy_policy.train()
        self.allocation_policy.train()
        self.execution_policy.train()

    def _set_eval_mode(self):
        """Set all networks to evaluation mode"""
        self.strategy_policy.eval()
        self.allocation_policy.eval()
        self.execution_policy.eval()

    def select_action(self, state: np.ndarray, step: int,
                      symbols: List[str], deterministic: bool = None):
        """
        Select action based on current mode - FIXED with dimension validation
        """
        # Determine if we should be deterministic based on mode
        if deterministic is None:
            deterministic = (self.mode != TradingMode.TRAINING)

        with torch.no_grad() if deterministic else torch.enable_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Level 1: Strategy (every 60 steps)
            if step % 60 == 0 or self.current_strategy is None:
                strategy_output = self._select_strategy(state_tensor, deterministic)
                self.current_strategy = strategy_output['probs']
                self.strategy_value = strategy_output['value']
                self._last_strategy_info = strategy_output['meta']

            # Level 2: Allocation (every 10 steps)
            if step % 10 == 0 or self.current_allocation is None:
                allocation_input = torch.FloatTensor(
                    np.concatenate([state, self.current_strategy])
                ).unsqueeze(0)

                allocation_output = self._select_allocation(allocation_input, deterministic)
                self.current_allocation = allocation_output['weights']
                self.allocation_value = allocation_output['value']
                self._last_allocation_info = allocation_output['meta']

            # CRITICAL: Validate dimensions before concatenation
            state_flat = state.flatten()
            strategy_flat = self.current_strategy.flatten()
            allocation_flat = self.current_allocation.flatten()

            # Log dimensions for debugging
            if step == 0:
                logging.info(f"Dimension check: state={state_flat.shape}, "
                             f"strategy={strategy_flat.shape}, "
                             f"allocation={allocation_flat.shape}")

            # Ensure all are 1D arrays
            if state_flat.ndim != 1 or strategy_flat.ndim != 1 or allocation_flat.ndim != 1:
                logging.error(f"Non-1D arrays detected! state:{state_flat.ndim}D, "
                              f"strategy:{strategy_flat.ndim}D, allocation:{allocation_flat.ndim}D")
                # Force to 1D
                state_flat = np.atleast_1d(state_flat).flatten()
                strategy_flat = np.atleast_1d(strategy_flat).flatten()
                allocation_flat = np.atleast_1d(allocation_flat).flatten()

            # Level 3: Execution (every step)
            try:
                execution_input = torch.FloatTensor(
                    np.concatenate([state_flat, strategy_flat, allocation_flat])
                ).unsqueeze(0)
            except ValueError as e:
                logging.error(f"Concatenation error: {e}")
                logging.error(f"  state shape: {state_flat.shape}")
                logging.error(f"  strategy shape: {strategy_flat.shape}")
                logging.error(f"  allocation shape: {allocation_flat.shape}")
                raise

            execution_output = self._select_execution(execution_input, deterministic)
            self.execution_value = execution_output['value']
            self._last_execution_info = execution_output['meta']

        # Prepare output
        weights = dict(zip(symbols + ['CASH'], self.current_allocation))

        return {
            'strategy': self._interpret_strategy(self.current_strategy),
            'allocation': weights,
            'execution': {
                'urgency': float(execution_output['urgency']),
                'aggressiveness': float(execution_output['aggressiveness']),
                'risk_reduction': float(execution_output['risk_reduction'])
            },
            '_meta': {
                'strategy': self._last_strategy_info,
                'allocation': self._last_allocation_info,
                'execution': self._last_execution_info
            }
        }

    def _select_strategy(self, state_tensor, deterministic):
        """Select strategy-level action"""
        strategy_logits, strategy_value = self.strategy_policy(state_tensor)

        if deterministic:
            # Deterministic: choose argmax
            strategy_idx = torch.argmax(strategy_logits, dim=1)
            strategy_probs = torch.softmax(strategy_logits, dim=1)
        else:
            # Stochastic: sample from categorical
            dist = D.Categorical(logits=strategy_logits)
            strategy_idx = dist.sample()
            strategy_probs = torch.softmax(strategy_logits, dim=1)

        # Calculate log probability for PPO
        dist = D.Categorical(logits=strategy_logits)
        strategy_logp = dist.log_prob(strategy_idx).item()

        return {
            'probs': strategy_probs.squeeze().detach().numpy(),
            'value': strategy_value.item(),
            'meta': {
                'idx': int(strategy_idx.item()),
                'logp': float(strategy_logp),
                'probs': strategy_probs.squeeze().detach().numpy().copy(),
                'logits': strategy_logits.detach().squeeze().numpy()
            }
        }

    def _select_allocation(self, allocation_input, deterministic):
        """Select allocation-level action - FIXED dimension handling"""
        allocation_logits, allocation_value = self.allocation_policy(allocation_input)

        if deterministic:
            allocation_probs = torch.softmax(allocation_logits, dim=1)
        else:
            # Use controlled exploration noise (UPDATED)
            noise = torch.randn_like(allocation_logits) * self.exploration_noise
            allocation_probs = torch.softmax(allocation_logits + noise, dim=1)

        # CRITICAL: Ensure correct dimensions
        allocation_weights = allocation_probs.squeeze(0).detach().numpy()  # Remove batch dim

        # Verify dimensions
        expected_dim = self.num_symbols + 1
        if allocation_weights.ndim == 0:  # Scalar
            logging.error(f"Allocation weights is scalar! Creating proper array.")
            allocation_weights = np.ones(expected_dim, dtype=np.float32) / expected_dim
        elif allocation_weights.shape[0] != expected_dim:
            logging.warning(f"Allocation weights wrong size: {allocation_weights.shape[0]} vs {expected_dim}")
            # Pad or truncate
            if allocation_weights.shape[0] < expected_dim:
                allocation_weights = np.pad(allocation_weights, (0, expected_dim - allocation_weights.shape[0]),
                                            'constant')
            else:
                allocation_weights = allocation_weights[:expected_dim]

        # Ensure it's 1D
        allocation_weights = allocation_weights.flatten()

        # Normalize to sum to 1
        allocation_weights = allocation_weights / (allocation_weights.sum() + 1e-10)

        allocation_logp = torch.log(allocation_probs + 1e-10).sum().item()

        return {
            'weights': allocation_weights,
            'value': allocation_value.item(),
            'meta': {
                'weights': allocation_weights.copy(),
                'logp': allocation_logp,
                'logits': allocation_logits.detach().squeeze(0).numpy()
            }
        }

    def _select_execution(self, execution_input, deterministic):
        """Select execution-level action"""
        exec_logits, execution_value = self.execution_policy(execution_input)
        exec_output = torch.sigmoid(exec_logits).squeeze()

        if not deterministic and self.mode == TradingMode.TRAINING:
            # Add exploration noise in training
            exec_noise = torch.randn_like(exec_output) * 0.01
            exec_output = torch.clamp(exec_output + exec_noise, 0, 1)

        exec_output_np = exec_output.detach().numpy()
        exec_logp = torch.log(torch.sigmoid(exec_logits) + 1e-10).sum().item()

        return {
            'urgency': exec_output_np[0],
            'aggressiveness': exec_output_np[1],
            'risk_reduction': exec_output_np[2],
            'value': execution_value.item(),
            'meta': {
                'logp': exec_logp,
                'raw': exec_logits.detach().squeeze().numpy()
            }
        }

    def store_transition(self, level: str, state, action, reward,
                         next_state, done, value, meta=None):
        """Store transition only in training mode"""
        if self.mode != TradingMode.TRAINING:
            return  # Don't store transitions when not training

        # Validate and adjust dimensions
        if level == 'strategy':
            target_dim = 60
            memory = self.strategy_memory
        elif level == 'allocation':
            target_dim = 65
            memory = self.allocation_memory
        elif level == 'execution':
            target_dim = 60 + 5 + self.num_symbols + 1
            memory = self.execution_memory
        else:
            raise ValueError(f"Unknown level: {level}")

        # Ensure correct dimensions
        state = self._ensure_dimension(state, target_dim)
        next_state = self._ensure_dimension(next_state, target_dim)

        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'value': value,
            'meta': meta
        }

        memory.append(transition)
        self.training_steps += 1

    def _ensure_dimension(self, arr, target_dim):
        """Ensure array has correct dimension"""
        if isinstance(arr, np.ndarray):
            arr = arr.flatten()
            if arr.shape[0] != target_dim:
                if arr.shape[0] < target_dim:
                    arr = np.pad(arr, (0, target_dim - arr.shape[0]), 'constant')
                else:
                    arr = arr[:target_dim]
        return arr

    def train(self):
        """Train all three policies (only in training mode)"""
        if self.mode != TradingMode.TRAINING:
            logging.warning("Train called but agent not in training mode")
            return {}

        metrics = {}

        # Train each level if enough data
        if len(self.strategy_memory) >= 100:
            metrics['strategy'] = self._train_policy(
                self.strategy_policy,
                self.strategy_optimizer,
                self.strategy_memory,
                'strategy'
            )

        if len(self.allocation_memory) >= 200:
            metrics['allocation'] = self._train_policy(
                self.allocation_policy,
                self.allocation_optimizer,
                self.allocation_memory,
                'allocation'
            )

        if len(self.execution_memory) >= 400:
            metrics['execution'] = self._train_policy(
                self.execution_policy,
                self.execution_optimizer,
                self.execution_memory,
                'execution'
            )

        self.episodes_trained += 1

        # Periodically clear old memories to prevent overfitting
        if self.episodes_trained % 100 == 0:
            self._clear_old_memories()

        return metrics

    def _clear_old_memories(self):
        """Clear oldest 20% of memories periodically"""
        for memory in [self.strategy_memory, self.allocation_memory, self.execution_memory]:
            if len(memory) > 1000:
                # Remove oldest 20%
                num_to_remove = len(memory) // 5
                for _ in range(num_to_remove):
                    memory.popleft()

    def _train_policy(self, policy, optimizer, memory, level_name):
        """PPO training with clipped objective"""
        batch_size = min(256, len(memory))
        indices = np.random.choice(len(memory), batch_size, replace=False)
        batch = [memory[i] for i in indices]

        # Prepare batch tensors
        states = torch.FloatTensor(np.array([t['state'] for t in batch], dtype=np.float32))
        rewards = torch.FloatTensor([t['reward'] for t in batch])
        dones = torch.FloatTensor([float(t['done']) for t in batch])
        old_values = torch.FloatTensor([t['value'] for t in batch])

        # Extract old log probabilities
        old_logps = []
        for t in batch:
            meta = t.get('meta', {})
            if isinstance(meta, dict):
                old_logps.append(float(meta.get('logp', 0.0)))
            else:
                old_logps.append(0.0)
        old_logps = torch.FloatTensor(old_logps)

        # Compute returns and advantages
        returns = self._compute_gae(rewards, old_values, dones)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        for epoch in range(5):
            logits, new_values = policy(states)

            # Calculate new log probabilities
            if logits.dim() == 2 and logits.size(1) > 1:
                if level_name == 'strategy':
                    dist = D.Categorical(logits=logits)
                    new_logps = torch.log(torch.softmax(logits, dim=1) + 1e-10).max(dim=1)[0]
                else:
                    probs = torch.softmax(logits, dim=1)
                    new_logps = torch.log(probs + 1e-10).sum(dim=1)
            else:
                new_logps = torch.log(torch.sigmoid(logits) + 1e-10).sum(dim=1)

            # PPO loss calculation
            ratio = torch.exp(new_logps - old_logps)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = ((new_values.squeeze() - returns) ** 2).mean()

            # Total loss
            total_loss = policy_loss + 0.5 * value_loss

            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

        return {
            'loss': float(total_loss.detach()),
            'value_loss': float(value_loss.detach()),
            'policy_loss': float(policy_loss.detach())
        }

    def _compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation returns"""
        returns = torch.zeros_like(rewards)
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            returns[t] = gae + values[t]

        return returns

    def _interpret_strategy(self, strategy_vector: np.ndarray) -> StrategyDecision:
        """Interpret strategy vector into decision"""
        regime_scores = strategy_vector[:3]
        regime = ['trending', 'ranging', 'volatile'][np.argmax(regime_scores)]
        sentiment = strategy_vector[3] - strategy_vector[4]

        if np.max(regime_scores) > 0.6:
            risk_mode = 'aggressive'
        elif np.max(regime_scores) > 0.4:
            risk_mode = 'moderate'
        else:
            risk_mode = 'conservative'

        return StrategyDecision(regime=regime, sentiment=sentiment, risk_mode=risk_mode)

    def save(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'strategy_policy': self.strategy_policy.state_dict(),
            'allocation_policy': self.allocation_policy.state_dict(),
            'execution_policy': self.execution_policy.state_dict(),
            'strategy_optimizer': self.strategy_optimizer.state_dict(),
            'allocation_optimizer': self.allocation_optimizer.state_dict(),
            'execution_optimizer': self.execution_optimizer.state_dict(),
            'num_symbols': self.num_symbols,
            'training_steps': self.training_steps,
            'episodes_trained': self.episodes_trained
        }, filepath)
        logging.info(f"Model saved to {filepath}")

    def load(self, filepath: str, mode: TradingMode = TradingMode.INFERENCE):
        """Load model checkpoint and set mode"""
        checkpoint = torch.load(filepath)
        self.strategy_policy.load_state_dict(checkpoint['strategy_policy'])
        self.allocation_policy.load_state_dict(checkpoint['allocation_policy'])
        self.execution_policy.load_state_dict(checkpoint['execution_policy'])

        if mode == TradingMode.TRAINING:
            # Only load optimizers if continuing training
            self.strategy_optimizer.load_state_dict(checkpoint['strategy_optimizer'])
            self.allocation_optimizer.load_state_dict(checkpoint['allocation_optimizer'])
            self.execution_optimizer.load_state_dict(checkpoint['execution_optimizer'])

        self.training_steps = checkpoint.get('training_steps', 0)
        self.episodes_trained = checkpoint.get('episodes_trained', 0)

        self.set_mode(mode)
        logging.info(f"Model loaded from {filepath} in {mode.value} mode")


class TemporalRLTradingSystem:
    """Enhanced main system with explicit mode management"""

    def __init__(self, config: Dict):
        self.config = config

        # Initialize components
        from autonomous_trading_bot.unified_data_provider import create_data_provider
        self.data_source = create_data_provider(config)

        self.portfolio_manager = PortfolioManagement(config)

        symbols = config.get('symbols_to_trade', [])

        # Create environment
        self.env = TemporalTradingEnvironment(
            config, self.data_source, self.portfolio_manager
        )

        # Create agent
        self.agent = TemporalAbstractionAgent(
            num_symbols=len(symbols),
            lr=config.get('learning_rate', 3e-4)
        )

        # Initialize baselines
        self.baselines = {
            'buy_and_hold': BuyAndHoldBaseline(config),
            'equal_weight': EqualWeightBaseline(config),
            'random': RandomBaseline(config)
        }

        # Training history
        self.training_history = []
        self.validation_history = []

        self.training_phases = config.get('training_phases', {
            'exploration': {
                'start': 0,
                'episodes': 5000,
                'exploration_noise': 0.3,
                'description': 'Explore all data segments',
                'validation_freq': 100
            },
            'refinement': {
                'start': 5000,
                'episodes': 3000,
                'exploration_noise': 0.1,
                'description': 'Refine on challenging segments',
                'validation_freq': 50
            },
            'optimization': {
                'start': 8000,
                'episodes': 2000,
                'exploration_noise': 0.01,
                'description': 'Verify generalization',
                'validation_freq': 25
            }
        })

        # Episode tracking (NEW)
        self.episode_segments = []
        self.segment_performance = defaultdict(list)

        logging.info(f"System initialized for {len(symbols)} symbols")

    @property
    def episode_config(self):
        """Access episode config from data source"""
        if hasattr(self.data_source, 'episode_config'):
            return self.data_source.episode_config
        # Fallback: create default config
        from autonomous_trading_bot.unified_data_provider import EpisodeConfig
        return EpisodeConfig()

    def train(self, num_episodes: int = 1000,
              validation_interval: int = None,
              checkpoint_interval: int = 50):
        """
        Enhanced training with unlimited episodes

        CRITICAL: num_episodes is NOT limited by data!

        Args:
            num_episodes: Total episodes (can be >> unique segments)
            validation_interval: Override auto validation (None = use phase default)
            checkpoint_interval: Episodes between checkpoints
        """

        # Set agent to training mode
        self.agent.set_mode(TradingMode.TRAINING)

        # Calculate data statistics (NEW)
        available_segments = self.data_source.episode_config.get_estimated_episodes_available('train')
        expected_reuse = num_episodes / max(1, available_segments)

        logger.info("\n" + "=" * 70)
        logger.info("UNLIMITED EPISODES TRAINING")
        logger.info("=" * 70)
        logger.info(f"Total episodes: {num_episodes:,}")
        logger.info(f"Unique segments: {available_segments}")
        logger.info(f"Expected reuse: {expected_reuse:.1f}x per segment")
        logger.info("=" * 70 + "\n")

        best_validation_score = -float('inf')
        patience = 50
        patience_counter = 0

        for episode in range(num_episodes):
            # Determine current phase (NEW)
            phase_info = self._get_training_phase(episode)

            # Set exploration noise (NEW)
            self.agent.set_exploration_noise(phase_info['exploration_noise'])

            # Get validation interval for this phase (NEW)
            val_interval = validation_interval or phase_info.get('validation_freq', 100)

            # Training episode
            metrics = self. train_episode()

            # Track segment info (NEW)
            if hasattr(self.data_source, 'episode_start_index'):
                segment_id = self.data_source.episode_start_index
                self.episode_segments.append(segment_id)
                self.segment_performance[segment_id].append(metrics['return'])
                metrics['segment_id'] = segment_id

            # Add phase info (NEW)
            metrics['phase'] = phase_info['name']
            metrics['exploration_noise'] = phase_info['exploration_noise']

            self.training_history.append(metrics)

            # Log progress (UPDATED)
            if episode % 100 == 0:
                self._log_training_progress(episode, metrics, phase_info)

            # Validation (UPDATED interval)
            if episode % val_interval == 0 and episode > 0:
                validation_metrics = self.validate()
                self.validation_history.append(validation_metrics)

                val_score = validation_metrics['vs_baselines_score']
                if val_score > best_validation_score:
                    best_validation_score = val_score
                    patience_counter = 0
                    self.agent.save('models/best_model.pth')
                    logger.info(f"✅ New best: {val_score:.4f}")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at episode {episode}")
                    break

            # Checkpointing
            if episode % checkpoint_interval == 0 and episode > 0:
                self.agent.save(f'models/checkpoint_ep{episode}.pth')
                self._log_diversity_report()

        # Final save
        self.agent.save('models/final_model.pth')
        self._log_diversity_report()

        return self.training_history

    def train_episode(self) -> Dict:
        """Single training episode"""
        self.portfolio_manager.reset_daily_values()
        # Ensure training mode
        self.agent.set_mode(TradingMode.TRAINING)
        mode = 'train'
        # Reset environment with mode
        state = self.env.reset(mode=mode)  # CHANGED
        episode_reward = 0
        step = 0
        initial_value = self.portfolio_manager.get_portfolio_value()

        # Episode loop
        done = False
        while not done:
            # Get action from agent
            action_dict = self.agent.select_action(
                state, step, self.env.symbols, deterministic=False
            )

            # Environment step
            next_state, reward, done, info = self.env.step(action_dict)

            # Store transitions for all levels
            self._store_hierarchical_transitions(
                state, action_dict, reward, next_state, done, info, step
            )

            state = next_state
            episode_reward += reward
            step += 1

        # Calculate final metrics
        final_value = self.portfolio_manager.get_portfolio_value()
        episode_return = (final_value - initial_value) / initial_value

        # Train the agent
        train_metrics = self.agent.train()

        # Evaluate against baselines on same data
        baseline_results = self._evaluate_baselines_on_episode()

        return {
            'episode': len(self.training_history) + 1,
            'steps': step,
            'reward': episode_reward,
            'return': episode_return,
            'initial_value': initial_value,
            'final_value': final_value,
            'baselines': baseline_results,
            'training': train_metrics
        }

    def _store_hierarchical_transitions(self, state, action_dict, reward,
                                        next_state, done, info, step):
        """Store transitions at appropriate temporal levels"""
        state_flat = state.flatten()
        next_state_flat = next_state.flatten()

        # Strategy level (every 60 steps)
        if step % 60 == 0:
            self.agent.store_transition(
                'strategy',
                state,
                action_dict['strategy'],
                info.get('strategy_reward', 0),
                next_state,
                done,
                self.agent.strategy_value,
                action_dict.get('_meta', {}).get('strategy')
            )

        # Allocation level (every 10 steps)
        if step % 10 == 0:
            strategy_flat = np.array(self.agent.current_strategy).flatten()
            allocation_state = np.concatenate([state_flat, strategy_flat])
            allocation_next = np.concatenate([next_state_flat, strategy_flat])

            self.agent.store_transition(
                'allocation',
                allocation_state,
                action_dict['allocation'],
                info.get('allocation_reward', 0),
                allocation_next,
                done,
                self.agent.allocation_value,
                action_dict.get('_meta', {}).get('allocation')
            )

        # Execution level (every step)
        strategy_flat = np.array(self.agent.current_strategy).flatten()
        allocation_flat = np.array(self.agent.current_allocation).flatten()
        execution_state = np.concatenate([state_flat, strategy_flat, allocation_flat])
        execution_next = np.concatenate([next_state_flat, strategy_flat, allocation_flat])

        self.agent.store_transition(
            'execution',
            execution_state,
            action_dict['execution'],
            reward,
            execution_next,
            done,
            self.agent.execution_value,
            action_dict.get('_meta', {}).get('execution')
        )

    def validate(self) -> Dict:
        """Validation with comparison to baselines"""
        # Set agent to evaluation mode
        self.agent.set_mode(TradingMode.EVALUATION)

        val_episodes = 5
        rl_returns = []
        baseline_returns = {name: [] for name in self.baselines}

        for _ in range(val_episodes):
            # RL agent evaluation - use mode='validation'
            rl_return = self._run_evaluation_episode(mode='validation')  # CHANGED
            rl_returns.append(rl_return)

            # Baselines
            for name, baseline in self.baselines.items():
                baseline_return = self._run_baseline_episode(baseline)  # CHANGED
                baseline_returns[name].append(baseline_return)

        # Calculate metrics
        avg_rl = np.mean(rl_returns)
        avg_baselines = {name: np.mean(returns) for name, returns in baseline_returns.items()}

        # Score: how much better than baselines
        vs_baselines_score = sum(
            1 if avg_rl > baseline_avg else 0
            for baseline_avg in avg_baselines.values()
        ) / len(avg_baselines)

        # Return agent to training mode
        self.agent.set_mode(TradingMode.TRAINING)

        return {
            'rl_return': avg_rl,
            'baselines': avg_baselines,
            'vs_baselines_score': vs_baselines_score
        }

    def _run_evaluation_episode(self, mode: str = 'validation') -> float:
        """Run single evaluation episode for RL agent"""
        state = self.env.reset(mode=mode)  # CHANGED
        initial_value = self.portfolio_manager.get_portfolio_value()
        step = 0
        done = False

        while not done:
            action_dict = self.agent.select_action(
                state, step, self.env.symbols, deterministic=True
            )
            state, _, done, _ = self.env.step(action_dict)
            step += 1

        final_value = self.portfolio_manager.get_portfolio_value()
        return (final_value - initial_value) / initial_value

    def _run_baseline_episode(self, baseline) -> float:
        """Run single episode for baseline strategy"""
        self.portfolio_manager.reset()
        state = self.env.reset()

        try:
            return baseline.execute(self.env, self.portfolio_manager)
        except Exception as e:
            logging.error(f"Baseline execution error: {e}")
            return 0.0

    def _evaluate_baselines_on_episode(self) -> Dict:
        """Evaluate baselines on same episode data"""
        episode_start_index = self.data_source.current_index
        results = {}

        for name, baseline in self.baselines.items():
            try:
                # Reset to same starting point
                self.data_source.current_index = episode_start_index
                self.portfolio_manager.reset()

                # Execute baseline
                baseline_return = baseline.execute(self.env, self.portfolio_manager)
                results[name] = baseline_return

            except Exception as e:
                logging.error(f"Baseline {name} error: {e}")
                results[name] = 0.0

        return results

    def should_deploy(self, eval_episodes: int = 50) -> bool:
        """Fixed deployment decision with proper baseline evaluation"""

        return False
        print("\n" + "=" * 80)
        print("DEPLOYMENT DECISION ANALYSIS")
        print("=" * 80)

        # Set to evaluation mode
        self.agent.set_mode(TradingMode.EVALUATION)

        # Evaluate RL agent
        rl_results = self.evaluate(eval_episodes, deterministic=True)

        # Evaluate baselines
        baseline_results = {}
        for name, baseline in self.baselines.items():
            returns = []
            for _ in range(eval_episodes):
                # Reset environment and portfolio
                self.portfolio_manager.reset()
                state = self.env.reset()

                if not self.data_source.cache_mode:
                    logging.warning(f"Cache not active for baseline {name}")
                    continue

                # Execute baseline strategy
                try:
                    ret = baseline.execute(self.env, self.portfolio_manager)
                    returns.append(ret)
                except Exception as e:
                    logging.error(f"Error in baseline {name}: {e}")
                    returns.append(0.0)

            baseline_results[name] = {
                'mean_return': np.mean(returns) if returns else 0.0,
                'std_return': np.std(returns) if returns else 0.0
            }

        # Extract metrics
        rl_mean = rl_results['mean_return']
        rl_sharpe = rl_results['mean_sharpe']
        rl_max_dd = rl_results['mean_max_drawdown']

        bnh_mean = baseline_results['buy_and_hold']['mean_return']
        ew_mean = baseline_results['equal_weight']['mean_return']

        # Deployment criteria
        criteria = {
            'beats_buy_and_hold': rl_mean > bnh_mean,
            'beats_equal_weight': rl_mean > ew_mean,
            'positive_sharpe': rl_sharpe > 0.5,
            'acceptable_drawdown': rl_max_dd < 0.15,
            'consistent': rl_results['std_return'] < 0.1
        }

        # Display results
        print(f"\nRL Agent Performance:")
        print(f"  Mean Return:     {rl_mean:>8.2%}")
        print(f"  Sharpe Ratio:    {rl_sharpe:>8.2f}")
        print(f"  Max Drawdown:    {rl_max_dd:>8.2%}")
        print(f"  Std of Returns:  {rl_results['std_return']:>8.2%}")

        print(f"\nBaseline Comparison:")
        print(f"  Buy-and-Hold:    {bnh_mean:>8.2%}  {'✓' if criteria['beats_buy_and_hold'] else '✗'}")
        print(f"  Equal-Weight:    {ew_mean:>8.2%}  {'✓' if criteria['beats_equal_weight'] else '✗'}")

        print(f"\nDeployment Criteria:")
        for criterion, passed in criteria.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {criterion.replace('_', ' ').title():<25} {status}")

        deploy = all(criteria.values())

        print(f"\n{'=' * 80}")
        if deploy:
            print("✓✓✓ RECOMMENDATION: DEPLOY ✓✓✓")
        else:
            print("✗✗✗ RECOMMENDATION: DO NOT DEPLOY ✗✗✗")
            print("\nFailed criteria need improvement before deployment.")
        print("=" * 80 + "\n")

        return deploy

    def deploy(self, model_path: str = 'models/best_model.pth'):
        """Deploy model for live/paper trading"""
        # Load best model in inference mode
        self.agent.load(model_path, mode=TradingMode.INFERENCE)

        logging.info("Model deployed for inference")
        logging.info(f"Mode: {self.agent.mode.value}")
        logging.info(f"Training steps completed: {self.agent.training_steps}")
        logging.info(f"Episodes trained: {self.agent.episodes_trained}")

        return self.agent

    def _get_training_phase(self, episode: int) -> Dict:
        """Determine current training phase"""
        for phase_name, phase_config in self.training_phases.items():
            phase_start = phase_config['start']
            phase_end = phase_start + phase_config['episodes']

            if phase_start <= episode < phase_end:
                return {'name': phase_name, **phase_config}

        # Default to last phase
        last_phase = list(self.training_phases.keys())[-1]
        return {'name': last_phase, **self.training_phases[last_phase]}

    def _log_training_progress(self, episode: int, metrics: Dict,
                               phase_info: Dict):
        """Log detailed training progress"""
        logger.info(f"\nEpisode {episode:,} - {phase_info['name'].upper()}")
        logger.info(f"  Phase: {phase_info['description']}")
        logger.info(f"  Exploration: {phase_info['exploration_noise']:.3f}")
        logger.info(f"  Return: {metrics['return']:.4f}")

        # Segment stats
        if 'segment_id' in metrics:
            segment_returns = self.segment_performance[metrics['segment_id']]
            if len(segment_returns) > 1:
                logger.info(f"  Segment visits: {len(segment_returns)}")
                logger.info(f"  Segment avg: {np.mean(segment_returns):.4f}")

        # Diversity stats
        if hasattr(self.data_source, 'get_episode_diversity_stats'):
            stats = self.data_source.get_episode_diversity_stats()
            logger.info(f"  Coverage: {stats['coverage']:.1%}")
            logger.info(f"  Avg reuse: {stats['avg_reuse']:.1f}x")

    def _log_diversity_report(self):
        """Log episode diversity statistics"""
        if not hasattr(self.data_source, 'get_episode_diversity_stats'):
            return

        stats = self.data_source.get_episode_diversity_stats()

        logger.info("\n" + "=" * 70)
        logger.info("EPISODE DIVERSITY REPORT")
        logger.info("=" * 70)
        logger.info(f"Episodes: {stats['total_episodes']}")
        logger.info(f"Unique segments: {stats['unique_segments_visited']} / {stats['total_possible_segments']}")
        logger.info(f"Coverage: {stats['coverage']:.1%}")
        logger.info(f"Avg reuse: {stats['avg_reuse']:.1f}x")
        logger.info(f"Max reuse: {stats['max_reuse']}x")
        logger.info("=" * 70 + "\n")

# Keep the PolicyNetwork class from original
class PolicyNetwork(nn.Module):
    """Shared policy architecture for all three levels"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.feature_weights = nn.Parameter(torch.ones(input_dim))

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.value_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x_weighted = x * torch.sigmoid(self.feature_weights)

        policy = self.network(x_weighted)
        value = self.value_head(x_weighted)
        return policy, value