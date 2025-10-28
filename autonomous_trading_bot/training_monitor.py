"""
Training Monitor - Track and visualize training progress
"""
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import logging

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """
    Monitor training progress with live plotting and checkpointing

    Usage:
        monitor = TrainingMonitor(save_dir='training_logs')

        for episode in range(num_episodes):
            # ... train episode ...

            monitor.log_episode({
                'episode': episode,
                'return': episode_return,
                'loss': loss,
                'reward': total_reward
            })

            # Auto-plot every 100 episodes
            if episode % 100 == 0:
                monitor.plot()

        monitor.save()
    """

    def __init__(self, save_dir: str = 'training_logs', window_size: int = 100):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

        self.window_size = window_size

        # Storage for all metrics
        self.metrics = {
            'episodes': [],
            'returns': [],
            'losses': [],
            'rewards': [],
            'portfolio_values': [],
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'win_rates': [],

            # Per-level metrics
            'strategy_losses': [],
            'allocation_losses': [],
            'execution_losses': [],

            # Validation metrics
            'val_returns': [],
            'val_episodes': [],

            # Baseline comparisons
            'vs_buy_hold': [],
            'vs_equal_weight': [],
        }

        # Moving averages
        self.moving_averages = {
            'returns': deque(maxlen=window_size),
            'losses': deque(maxlen=window_size),
        }

        logger.info(f"TrainingMonitor initialized - saving to {self.save_dir}")

    def log_episode(self, metrics: Dict):
        """
        Log metrics for one episode

        Args:
            metrics: Dict with keys like 'episode', 'return', 'loss', etc.
        """
        episode = metrics.get('episode', len(self.metrics['episodes']))
        self.metrics['episodes'].append(episode)

        # Core metrics
        if 'return' in metrics:
            ret = metrics['return']
            self.metrics['returns'].append(ret)
            self.moving_averages['returns'].append(ret)

        if 'loss' in metrics:
            loss = metrics['loss']
            self.metrics['losses'].append(loss)
            self.moving_averages['losses'].append(loss)

        # Optional metrics
        for key in ['reward', 'portfolio_value', 'sharpe_ratio', 'max_drawdown',
                    'win_rate', 'strategy_loss', 'allocation_loss', 'execution_loss']:
            if key in metrics:
                self.metrics[f"{key}s"].append(metrics[key])

        # Baseline comparisons
        if 'baselines' in metrics:
            baselines = metrics['baselines']
            if 'buy_and_hold' in baselines:
                self.metrics['vs_buy_hold'].append(
                    metrics.get('return', 0) - baselines['buy_and_hold']
                )
            if 'equal_weight' in baselines:
                self.metrics['vs_equal_weight'].append(
                    metrics.get('return', 0) - baselines['equal_weight']
                )

    def log_validation(self, episode: int, val_return: float):
        """Log validation metrics"""
        self.metrics['val_episodes'].append(episode)
        self.metrics['val_returns'].append(val_return)

    def get_smoothed(self, key: str, window: int = None) -> List[float]:
        """Get smoothed version of a metric"""
        if window is None:
            window = self.window_size

        values = self.metrics.get(key, [])
        if not values:
            return []

        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(values[start:i + 1]))

        return smoothed

    def plot(self, save: bool = True, show: bool = False):
        """
        Create comprehensive training plots

        Args:
            save: Save plots to disk
            show: Display plots (blocks execution)
        """
        if not self.metrics['episodes']:
            logger.warning("No metrics to plot yet")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')

        episodes = self.metrics['episodes']

        # Plot 1: Returns
        ax = axes[0, 0]
        if self.metrics['returns'] and len(self.metrics['returns']) == len(episodes):
            ax.plot(episodes, self.metrics['returns'], alpha=0.3, label='Episode Return')
            smoothed = self.get_smoothed('returns')
            if smoothed:
                ax.plot(episodes[:len(smoothed)], smoothed, linewidth=2,
                        label=f'{self.window_size}-ep Average')

            if self.metrics['val_returns']:
                ax.scatter(self.metrics['val_episodes'], self.metrics['val_returns'],
                           color='red', s=50, zorder=5, label='Validation')

            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Return')
            ax.set_title('Episode Returns')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No return data yet', ha='center', va='center')
            ax.set_title('Episode Returns')

        # Plot 2: Losses
        ax = axes[0, 1]
        if self.metrics['losses']:
            # Only plot losses where we have data
            loss_episodes = episodes[:len(self.metrics['losses'])]
            ax.plot(loss_episodes, self.metrics['losses'], alpha=0.3, label='Loss')

            smoothed = self.get_smoothed('losses')
            if smoothed:
                ax.plot(loss_episodes[:len(smoothed)], smoothed, linewidth=2,
                        label=f'{self.window_size}-ep Average')

            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No loss data yet', ha='center', va='center')
            ax.set_title('Training Loss')

        # Plot 3: Portfolio Value
        ax = axes[0, 2]
        if self.metrics['portfolio_values']:
            pv_episodes = episodes[:len(self.metrics['portfolio_values'])]
            ax.plot(pv_episodes, self.metrics['portfolio_values'], linewidth=2)
            ax.axhline(y=100000, color='gray', linestyle='--', alpha=0.5, label='Initial')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Portfolio Value ($)')
            ax.set_title('Portfolio Value Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No portfolio data yet', ha='center', va='center')
            ax.set_title('Portfolio Value')

        # Plot 4: Sharpe Ratio
        ax = axes[1, 0]
        if self.metrics['sharpe_ratios']:
            sr_episodes = episodes[:len(self.metrics['sharpe_ratios'])]
            smoothed = self.get_smoothed('sharpe_ratios', window=50)
            if smoothed:
                ax.plot(sr_episodes[:len(smoothed)], smoothed, linewidth=2)
            ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Target')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Sharpe Ratio')
            ax.set_title('Sharpe Ratio (50-ep avg)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Sharpe data yet', ha='center', va='center')
            ax.set_title('Sharpe Ratio')

        # Plot 5: vs Baselines
        ax = axes[1, 1]
        has_baseline_data = False

        if self.metrics['vs_buy_hold']:
            bh_episodes = episodes[:len(self.metrics['vs_buy_hold'])]
            smoothed = self.get_smoothed('vs_buy_hold')
            if smoothed:
                ax.plot(bh_episodes[:len(smoothed)], smoothed,
                        label='vs Buy & Hold', linewidth=2)
                has_baseline_data = True

        if self.metrics['vs_equal_weight']:
            ew_episodes = episodes[:len(self.metrics['vs_equal_weight'])]
            smoothed = self.get_smoothed('vs_equal_weight')
            if smoothed:
                ax.plot(ew_episodes[:len(smoothed)], smoothed,
                        label='vs Equal Weight', linewidth=2)
                has_baseline_data = True

        if has_baseline_data:
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Excess Return')
            ax.set_title('Performance vs Baselines')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No baseline data yet', ha='center', va='center')
            ax.set_title('Performance vs Baselines')

        # Plot 6: Win Rate
        ax = axes[1, 2]
        if self.metrics['win_rates']:
            wr_episodes = episodes[:len(self.metrics['win_rates'])]
            smoothed = self.get_smoothed('win_rates', window=50)
            if smoothed:
                ax.plot(wr_episodes[:len(smoothed)], smoothed, linewidth=2)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Win Rate')
            ax.set_title('Win Rate (50-ep avg)')
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No win rate data yet', ha='center', va='center')
            ax.set_title('Win Rate')

        plt.tight_layout()

        if save:
            plot_path = self.save_dir / 'training_progress.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {plot_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def save(self, filename: str = 'training_metrics.json'):
        """Save metrics to JSON"""
        save_path = self.save_dir / filename

        # Convert numpy types to native Python for JSON
        metrics_json = {}
        for key, values in self.metrics.items():
            metrics_json[key] = [
                float(v) if isinstance(v, (np.floating, np.integer)) else v
                for v in values
            ]

        with open(save_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)

        logger.info(f"Metrics saved to {save_path}")

    def load(self, filename: str = 'training_metrics.json'):
        """Load metrics from JSON"""
        load_path = self.save_dir / filename

        if not load_path.exists():
            logger.warning(f"Metrics file not found: {load_path}")
            return

        with open(load_path, 'r') as f:
            self.metrics = json.load(f)

        logger.info(f"Metrics loaded from {load_path}")

    def print_summary(self, last_n: int = 100):
        """Print training summary"""
        if not self.metrics['episodes']:
            print("No training data yet")
            return

        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)

        total_episodes = len(self.metrics['episodes'])
        print(f"\nTotal Episodes: {total_episodes}")

        if self.metrics['returns']:
            recent_returns = self.metrics['returns'][-last_n:]
            print(f"\nReturns (last {min(last_n, len(recent_returns))} episodes):")
            print(f"  Mean:   {np.mean(recent_returns):>8.4f}")
            print(f"  Median: {np.median(recent_returns):>8.4f}")
            print(f"  Std:    {np.std(recent_returns):>8.4f}")
            print(f"  Min:    {np.min(recent_returns):>8.4f}")
            print(f"  Max:    {np.max(recent_returns):>8.4f}")

        if self.metrics['losses']:
            recent_losses = self.metrics['losses'][-last_n:]
            print(f"\nLosses (last {min(last_n, len(recent_losses))} episodes):")
            print(f"  Mean:   {np.mean(recent_losses):>8.4f}")

        if self.metrics['val_returns']:
            print(f"\nValidation Returns:")
            print(f"  Latest: {self.metrics['val_returns'][-1]:>8.4f}")
            print(f"  Best:   {np.max(self.metrics['val_returns']):>8.4f}")

        print("=" * 70 + "\n")


# ============================================================================
# CHECKPOINT MANAGER - Save/Load Agent with Training State
# ============================================================================

class CheckpointManager:
    """
    Manage model checkpoints and training state

    Usage:
        checkpoint_mgr = CheckpointManager('checkpoints')

        # Save
        checkpoint_mgr.save(agent, episode=1000, metrics={'return': 0.05})

        # Load and continue training
        start_episode = checkpoint_mgr.load(agent, 'checkpoint_ep1000.pth')
    """

    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def save(self, agent, episode: int, metrics: Dict = None,
             monitor: TrainingMonitor = None, is_best: bool = False):
        """
        Save complete checkpoint

        Args:
            agent: Your TemporalAbstractionAgent
            episode: Current episode number
            metrics: Optional metrics dict
            monitor: Optional TrainingMonitor to save
            is_best: If True, also save as 'best_model.pth'
        """
        import torch

        # Unwrap CachedAgent if needed
        if hasattr(agent, 'agent'):
            actual_agent = agent.agent
        else:
            actual_agent = agent

        checkpoint = {
            'episode': episode,
            'strategy_policy': actual_agent.strategy_policy.state_dict(),
            'allocation_policy': actual_agent.allocation_policy.state_dict(),
            'execution_policy': actual_agent.execution_policy.state_dict(),
            'strategy_optimizer': actual_agent.strategy_optimizer.state_dict(),
            'allocation_optimizer': actual_agent.allocation_optimizer.state_dict(),
            'execution_optimizer': actual_agent.execution_optimizer.state_dict(),
            'num_symbols': actual_agent.num_symbols,
            'training_steps': actual_agent.training_steps,
            'episodes_trained': actual_agent.episodes_trained,
            'mode': actual_agent.mode.value if hasattr(actual_agent.mode, 'value') else str(actual_agent.mode),
        }

        if metrics:
            checkpoint['metrics'] = metrics

        # Save main checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_ep{episode}.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Save as best if needed
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")

        # Save monitor
        if monitor:
            monitor.save(f'metrics_ep{episode}.json')

    def load(self, agent, checkpoint_path: str = None, load_optimizers: bool = True) -> int:
        """
        Load checkpoint and return starting episode

        Args:
            agent: Your TemporalAbstractionAgent
            checkpoint_path: Path to checkpoint (None = load latest)
            load_optimizers: Whether to restore optimizer state

        Returns:
            Starting episode number (to continue training)
        """
        import torch

        # Find checkpoint
        if checkpoint_path is None:
            # Load latest checkpoint
            checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_ep*.pth'))
            if not checkpoints:
                logger.warning("No checkpoints found")
                return 0
            checkpoint_path = checkpoints[-1]
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return 0

        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)

        # Unwrap CachedAgent if needed
        if hasattr(agent, 'agent'):
            actual_agent = agent.agent
        else:
            actual_agent = agent

        # Load model states
        actual_agent.strategy_policy.load_state_dict(checkpoint['strategy_policy'])
        actual_agent.allocation_policy.load_state_dict(checkpoint['allocation_policy'])
        actual_agent.execution_policy.load_state_dict(checkpoint['execution_policy'])

        # Load optimizer states (for continuing training)
        if load_optimizers:
            actual_agent.strategy_optimizer.load_state_dict(checkpoint['strategy_optimizer'])
            actual_agent.allocation_optimizer.load_state_dict(checkpoint['allocation_optimizer'])
            actual_agent.execution_optimizer.load_state_dict(checkpoint['execution_optimizer'])

        # Restore training state
        actual_agent.training_steps = checkpoint.get('training_steps', 0)
        actual_agent.episodes_trained = checkpoint.get('episodes_trained', 0)

        episode = checkpoint.get('episode', 0)

        logger.info(f"✓ Checkpoint loaded: episode {episode}")
        logger.info(f"  Training steps: {actual_agent.training_steps}")
        logger.info(f"  Episodes trained: {actual_agent.episodes_trained}")

        if 'metrics' in checkpoint:
            logger.info(f"  Last metrics: {checkpoint['metrics']}")

        return episode

    def list_checkpoints(self):
        """List all available checkpoints"""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_ep*.pth'))

        print("\nAvailable Checkpoints:")
        print("-" * 50)

        for cp in checkpoints:
            print(f"  {cp.name}")

        print("-" * 50)
        print(f"Total: {len(checkpoints)} checkpoints\n")

        return checkpoints

    def find_latest_checkpoint(self):
        """
        Find the most recent checkpoint automatically

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist
        """
        import logging
        logger = logging.getLogger(__name__)

        checkpoints = sorted(
            self.checkpoint_dir.glob('checkpoint_ep*.pth'),
            key=lambda p: int(p.stem.split('ep')[1])  # Sort by episode number
        )

        if checkpoints:
            latest = checkpoints[-1]
            logger.debug(f"Found latest checkpoint: {latest}")
            return str(latest)

        logger.debug("No existing checkpoints found")
        return None

    def auto_resume(self, agent) -> int:
        """
        Automatically resume from latest checkpoint if available

        Returns:
            Starting episode (0 if no checkpoint, or last episode + 1)
        """
        latest = self.find_latest_checkpoint()

        if latest:
            logger.info(f"🔄 Auto-resuming from: {latest}")
            return self.load(agent, latest)
        else:
            logger.info("🆕 Starting fresh training (no checkpoints found)")
            return 0

# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def setup_monitoring(save_dir: str = 'training_logs') -> tuple:
    """
    Quick setup for monitoring and checkpointing

    Returns:
        (monitor, checkpoint_manager)

    Usage:
        monitor, checkpoints = setup_monitoring()
    """
    monitor = TrainingMonitor(save_dir=save_dir)
    checkpoint_mgr = CheckpointManager(checkpoint_dir=f'{save_dir}/checkpoints')

    return monitor, checkpoint_mgr