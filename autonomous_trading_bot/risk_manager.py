"""
Fix for Risk Manager - Better Limits for RL Training

Apply these changes to risk_manager.py
"""
from typing import Dict, Optional
import logging

from autonomous_trading_bot.logging_setup import setup_logging

setup_logging(level="ERROR")
# ============================================================================
# SOLUTION 1: Update config.json with RL-friendly limits
# ============================================================================

RL_FRIENDLY_CONFIG = {
    # Relaxed limits for training
    "max_drawdown_percent": 25.0,  # Was: 10.0
    "daily_loss_limit_percent": 15.0,  # Was: 5.0

    # Keep position limits reasonable
    "max_position_size_percent": 20,
    "max_open_positions": 20,

    # Portfolio risk
    "max_portfolio_risk_percent": 10,  # Was: 5

    # For live trading, use stricter limits
    "live_trading_limits": {
        "max_drawdown_percent": 10.0,
        "daily_loss_limit_percent": 5.0,
        "max_portfolio_risk_percent": 5
    }
}


# ============================================================================
# SOLUTION 2: Modify RiskManager to be training-aware
# ============================================================================

class RiskManager:
    """Enhanced RiskManager with training mode"""

    def __init__(self, config):
        self.config = config
        self.mode = config.get('mode', 'live')

        # Different limits for training vs live
        if self.mode in ['simulation', 'rl_training']:
            # Relaxed limits for training
            self.max_drawdown = config.get('training_max_drawdown_percent', 25.0) / 100
            self.daily_loss_limit = config.get('training_daily_loss_limit_percent', 15.0) / 100
            self.max_portfolio_risk = config.get('training_max_portfolio_risk_percent', 10.0) / 100
        else:
            # Strict limits for live trading
            self.max_drawdown = config.get('max_drawdown_percent', 10.0) / 100
            self.daily_loss_limit = config.get('daily_loss_limit_percent', 5.0) / 100
            self.max_portfolio_risk = config.get('max_portfolio_risk_percent', 5.0) / 100

        # Position limits (same for both)
        self.max_position_size = config.get('max_position_size_percent', 20) / 100
        self.max_positions = config.get('max_open_positions', 20)

        # Rate limiting
        self.max_order_frequency = config.get('max_orders_per_minute', 60)
        self.min_profit_threshold = config.get('min_profit_threshold', 0.001)

        # Track state
        self.order_timestamps = []

        # Track violations (instead of logging every time)
        self.violation_counts = {
            'drawdown': 0,
            'daily_loss': 0,
            'position_size': 0,
            'correlation': 0
        }
        self.last_violation_log = {}

        logging.info(f"RiskManager initialized in {self.mode} mode")
        logging.info(f"  Max drawdown: {self.max_drawdown * 100:.1f}%")
        logging.info(f"  Daily loss limit: {self.daily_loss_limit * 100:.1f}%")

    def _check_daily_loss_limit(self, portfolio_manager) -> bool:
        """Check daily loss limit with smart logging"""
        if not portfolio_manager:
            return True

        current_value = portfolio_manager.get_portfolio_value()
        daily_start_value = getattr(portfolio_manager, 'daily_start_value', current_value)

        if daily_start_value <= 0:
            return True

        daily_loss = (daily_start_value - current_value) / daily_start_value

        if daily_loss > self.daily_loss_limit:
            self.violation_counts['daily_loss'] += 1

            # Only log every 100 violations or if first time
            if self.violation_counts['daily_loss'] % 100 == 1:
                logging.warning(
                    f"Daily loss limit exceeded: {daily_loss * 100:.2f}% "
                    f"(limit: {self.daily_loss_limit * 100:.1f}%) "
                    f"[{self.violation_counts['daily_loss']} violations]"
                )

            # In training mode, DON'T halt trading - just warn
            if self.mode in ['simulation', 'rl_training']:
                return True  # Allow trade to continue
            else:
                return False  # Halt in live trading

        return True

    def _check_drawdown_limit(self, portfolio_manager) -> bool:
        """Check drawdown limit with smart logging"""
        if not portfolio_manager:
            return True

        current_value = portfolio_manager.get_portfolio_value()
        peak_value = getattr(portfolio_manager, 'peak_portfolio_value', current_value)

        if peak_value <= 0:
            return True

        drawdown = (peak_value - current_value) / peak_value

        if drawdown > self.max_drawdown:
            self.violation_counts['drawdown'] += 1

            # Only log every 100 violations or if first time
            if self.violation_counts['drawdown'] % 100 == 1:
                logging.warning(
                    f"Max drawdown exceeded: {drawdown * 100:.2f}% "
                    f"(limit: {self.max_drawdown * 100:.1f}%) "
                    f"[{self.violation_counts['drawdown']} violations]"
                )

            # In training mode, DON'T halt trading - just track
            if self.mode in ['simulation', 'rl_training']:
                return True  # Allow trade to continue
            else:
                return False  # Halt in live trading

        return True

    def validate_trade(self, signal: Dict, portfolio_manager, market_data: Dict) -> Optional[Dict]:
        """Validate trade with training-aware risk checks"""

        # Emergency stop (always respected)
        if self._check_emergency_conditions():
            logging.error("Emergency stop active - rejecting all trades")
            return None

        # Order frequency (always respected)
        if signal.get('strategy_name') == 'hft_ensemble':
            if not self._validate_order_frequency():
                logging.debug("Order frequency limit exceeded")
                return None

        # Position size validation (always respected)
        validated_signal = self._validate_position_size(signal, portfolio_manager)
        if not validated_signal:
            return None

        # Risk limit checks (training-aware)
        if not self._check_daily_loss_limit(portfolio_manager):
            return None

        if not self._check_drawdown_limit(portfolio_manager):
            return None

        # Correlation check (optional in training)
        if self.mode not in ['simulation', 'rl_training']:
            if not self._check_correlation_risk(signal['symbol'], portfolio_manager):
                logging.warning(f"Correlation risk too high for {signal['symbol']}")
                return None

        return validated_signal

    def get_violation_summary(self) -> Dict:
        """Get summary of risk violations"""
        return {
            'mode': self.mode,
            'violations': self.violation_counts.copy(),
            'limits': {
                'max_drawdown': self.max_drawdown * 100,
                'daily_loss_limit': self.daily_loss_limit * 100,
                'max_portfolio_risk': self.max_portfolio_risk * 100
            }
        }

    def reset_violation_counts(self):
        """Reset violation tracking (call at start of new episode)"""
        self.violation_counts = {k: 0 for k in self.violation_counts}

    # ... keep all other existing methods unchanged ...

    def _check_emergency_conditions(self) -> bool:
        """Check for emergency stop conditions"""
        import os
        return os.path.exists("STOP_TRADING")

    def _validate_order_frequency(self) -> bool:
        """Validate order frequency"""
        from datetime import datetime, timedelta

        current_time = datetime.now()

        # Remove old timestamps
        self.order_timestamps = [
            ts for ts in self.order_timestamps
            if (current_time - ts).total_seconds() < 60
        ]

        if len(self.order_timestamps) >= self.max_order_frequency:
            return False

        self.order_timestamps.append(current_time)
        return True

    def _validate_position_size(self, signal: Dict, portfolio_manager) -> Optional[Dict]:
        """Validate position size"""
        if not portfolio_manager:
            return signal

        portfolio_value = portfolio_manager.get_portfolio_value()
        max_position_value = portfolio_value * self.max_position_size

        position_value = signal['quantity'] * signal['price']

        if position_value > max_position_value:
            adjusted_quantity = int(max_position_value / signal['price'])
            if adjusted_quantity < 1:
                return None

            signal['quantity'] = adjusted_quantity
            logging.debug(f"Position size adjusted for {signal['symbol']}: {adjusted_quantity}")

        # Check cash for buy orders
        if signal['action'] == 'buy':
            available_cash = portfolio_manager.get_available_cash()
            if position_value > available_cash:
                adjusted_quantity = int(available_cash / signal['price'])
                if adjusted_quantity < 1:
                    return None
                signal['quantity'] = adjusted_quantity

        return signal

    def _check_correlation_risk(self, symbol: str, portfolio_manager) -> bool:
        """Simplified correlation check"""
        # In training, be more permissive
        if self.mode in ['simulation', 'rl_training']:
            return True

        # For live trading, implement proper correlation checks
        return True


# ============================================================================
# SOLUTION 3: Add training-specific config to config.json
# ============================================================================

CONFIG_ADDITIONS = """
Add to your config.json:

{
  ... existing config ...

  "training_max_drawdown_percent": 25.0,
  "training_daily_loss_limit_percent": 15.0,
  "training_max_portfolio_risk_percent": 10.0,

  "risk_logging": {
    "log_every_n_violations": 100,
    "track_violations": true
  }
}
"""


# ============================================================================
# SOLUTION 4: Add violation reporting to training
# ============================================================================

def report_risk_violations(system):
    """
    Add this to your training loop to report violations periodically
    Call every 1000 episodes or so
    """
    if hasattr(system.env, 'risk_manager'):
        summary = system.env.risk_manager.get_violation_summary()

        print("\n" + "=" * 70)
        print("RISK VIOLATION SUMMARY")
        print("=" * 70)
        print(f"Mode: {summary['mode']}")
        print(f"\nViolations:")
        for violation_type, count in summary['violations'].items():
            print(f"  {violation_type}: {count}")
        print(f"\nLimits:")
        for limit_name, value in summary['limits'].items():
            print(f"  {limit_name}: {value:.1f}%")
        print("=" * 70 + "\n")


# ============================================================================
# SOLUTION 5: Quick patch for immediate fix
# ============================================================================

QUICK_FIX = """
In portfolio_management.py, modify _check_risk_limits():

OLD:
    if current_drawdown > self.max_drawdown_percent:
        logging.critical(f"MAX DRAWDOWN LIMIT HIT! Current drawdown: {current_drawdown:.2f}%")

NEW:
    if current_drawdown > self.max_drawdown_percent:
        # Only log once every 100 violations
        if not hasattr(self, '_drawdown_violations'):
            self._drawdown_violations = 0
        self._drawdown_violations += 1

        if self._drawdown_violations % 100 == 1:
            logging.warning(f"Drawdown: {current_drawdown:.2f}% (limit: {self.max_drawdown_percent:.1f}%) "
                          f"[violation #{self._drawdown_violations}]")

Same for daily loss limit.
"""


# ============================================================================
# EXAMPLE: Integration with training
# ============================================================================

def train_with_risk_reporting(system, num_episodes=10000):
    """Example training loop with risk reporting"""

    for episode in range(num_episodes):
        # Train episode
        metrics = system.train_episode()

        # Reset violation counts each episode
        if hasattr(system.env, 'risk_manager'):
            system.env.risk_manager.reset_violation_counts()

        # Report every 1000 episodes
        if episode % 1000 == 0 and episode > 0:
            report_risk_violations(system)

    # Final report
    report_risk_violations(system)


if __name__ == "__main__":
    print("Risk Manager Fix for RL Training")
    print("\nChoose your solution:")
    print("1. Update config.json with relaxed limits (EASIEST)")
    print("2. Modify RiskManager class (BEST)")
    print("3. Quick patch in portfolio_management.py (FASTEST)")
    print("\nRecommendation: Use Solution 1 + 2 for best results")