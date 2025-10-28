import json
import os
from typing import Dict

def load_config(config_file='config.json'):
    """Loads configuration from a JSON file."""
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        return None
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(config, config_file='config.json'):
    """Saves configuration to a JSON file."""
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def _deep_get(d: Dict, path: str, default=None):
    cur = d
    for part in path.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def deep_get(cfg: Dict, path: str, default=None):
    """Public helper for safe nested access: deep_get(cfg, 'rl_config.rl_model_path')"""
    return _deep_get(cfg, path, default)

def get_encrypted_keys():
    """Load API keys from environment variables"""
    api_key = os.environ.get('ALPACA_API_KEY')
    api_secret = os.environ.get('ALPACA_API_SECRET')

    if not api_key or not api_secret:
        raise ValueError("API keys not found in environment variables")

    return api_key, api_secret

# Example default configuration (can be overridden by config.json)
default_config = {
    "api_key": "YOUR_ALPACA_API_KEY",
    "api_secret": "YOUR_ALPACA_API_SECRET",
    "base_url": "https://paper-api.alpaca.markets", # Use paper trading for development
    "data_url": "https://data.alpaca.markets",
    "trading_loop_interval": 60, # seconds
    "initial_capital": 10000.0,
    "risk_per_trade_percent": 1.0, # 1% of capital per trade
    "max_open_positions": 5,
    "max_drawdown_percent": 25.0, # Max 10% drawdown from peak equity
    "daily_loss_limit_percent": 15.0, # Max 5% loss in a single day
    "max_portfolio_risk_percent": 10.0,
    "rebalance_threshold_value": 100, # Rebalance if difference is more than $100
    "strtegy": {
        "type": "momentum", # or "hft", "mean_reversion"
        "params": {
            "lookback_period": 20, # for momentum
            "sma_short": 10,
            "sma_long": 50,
            "hft_threshold": 0.001 # for HFT
        }
    },
    "symbols_to_trade": ["AAPL", "MSFT", "GOOGL"]
}

# Create a default config.json if it doesn't exist
if not os.path.exists(os.path.join(os.path.dirname(__file__), 'config.json')):
    with open(os.path.join(os.path.dirname(__file__), 'config.json'), 'w') as f:
        json.dump(default_config, f, indent=4)


