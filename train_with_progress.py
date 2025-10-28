"""
Enhanced Training Script with Progress Bars and Reduced Verbosity
"""
import logging
import time
from pathlib import Path
from tqdm import tqdm
from autonomous_trading_bot.config import load_config
from autonomous_trading_bot.unified_data_provider import EpisodeConfig
from autonomous_trading_bot.temporal_rl_system import TemporalRLTradingSystem
from autonomous_trading_bot.logging_setup import setup_logging

# Import optimizations and monitoring
from targeted_optimizations import optimize
from training_monitor import setup_monitoring


class QuietLogger:
    """Context manager to temporarily reduce logging verbosity"""

    def __init__(self, level=logging.WARNING):
        self.level = level
        self.previous_levels = {}

    def __enter__(self):
        # Save current levels and set to WARNING
        for name in ['autonomous_trading_bot', 'matplotlib', 'yfinance']:
            logger = logging.getLogger(name)
            self.previous_levels[name] = logger.level
            logger.setLevel(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous levels
        for name, level in self.previous_levels.items():
            logging.getLogger(name).setLevel(level)


def train_with_progress(
        num_episodes: int = 10000,
        checkpoint_interval: int = 50,
        plot_interval: int = 100,
        validation_interval: int = 50,
        resume_from: str = None,
        auto_resume: bool = True,
        verbose: bool = False
):
    """
    Train with progress bar and minimal logging

    Args:
        num_episodes: Total episodes to train
        checkpoint_interval: Save checkpoint every N episodes
        plot_interval: Update plots every N episodes
        validation_interval: Run validation every N episodes
        resume_from: Path to checkpoint file to resume training
        auto_resume: Automatically resume from latest checkpoint
        verbose: Enable detailed logging
    """

    # Setup logging
    log_level = "INFO" if verbose else "WARNING"
    setup_logging(level="ERROR")


    # Suppress matplotlib and other noisy loggers
    if not verbose:
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('yfinance').setLevel(logging.ERROR)

    print("\n" + "=" * 70)
    print("TRAINING WITH PROGRESS TRACKING")
    print("=" * 70)

    # Setup monitoring
    monitor, checkpoint_mgr = setup_monitoring('training_logs')

    # Load config
    config = load_config('config.json')

    # Create episode config
    episode_config = EpisodeConfig.from_interval(
        interval='1Day',
        train_start='2022-01-01',
        train_end='2023-03-31',
        val_start='2023-04-01',
        val_end='2023-06-30',
        test_start='2023-07-01',
        test_end='2023-08-31'
    )
    config['episode_config'] = episode_config

    # Initialize system with quiet logging
    print("\n🔧 Initializing system...")
    with QuietLogger():
        system = TemporalRLTradingSystem(config)
        optimize(system, skip_compile=True)

    # Determine starting episode
    start_episode = 0

    if resume_from:
        print(f"📂 Resuming from: {resume_from}")
        start_episode = checkpoint_mgr.load(system.agent, resume_from)
        monitor.load()

    elif auto_resume:
        print("🔍 Checking for checkpoints...")
        latest = checkpoint_mgr.find_latest_checkpoint()
        if latest:
            start_episode = checkpoint_mgr.load(system.agent, latest)
            monitor.load()
            print(f"✅ Resuming from episode {start_episode}")
        else:
            print("🆕 Starting fresh training")

    print(f"\n📊 Training: episodes {start_episode} → {num_episodes}")
    print(f"💾 Checkpoints every {checkpoint_interval} episodes")
    print(f"📈 Validation every {validation_interval} episodes")
    print("=" * 70 + "\n")

    # Training loop with progress bar
    training_start = time.time()
    best_val_return = float('-inf')
    patience_counter = 0
    patience = 50

    # Metrics for progress bar
    recent_returns = []
    recent_losses = []

    try:
        # Create progress bar
        pbar = tqdm(
            range(start_episode, num_episodes),
            initial=start_episode,
            total=num_episodes,
            desc="Training",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )

        for episode in pbar:
            # Train one episode (with reduced logging)
            with QuietLogger(logging.ERROR if not verbose else logging.INFO):
                metrics = system.train_episode()

            # Extract key metrics
            episode_return = metrics.get('return', 0)
            episode_loss = 0
            if 'training' in metrics and 'execution' in metrics['training']:
                episode_loss = metrics['training']['execution'].get('loss', 0)

            # Track recent performance
            recent_returns.append(episode_return)
            if episode_loss > 0:
                recent_losses.append(episode_loss)

            # Keep only last 100
            if len(recent_returns) > 100:
                recent_returns.pop(0)
            if len(recent_losses) > 100:
                recent_losses.pop(0)

            # Log to monitor (silent)
            episode_metrics = {
                'episode': episode,
                'return': episode_return,
                'reward': metrics.get('reward', 0),
                'initial_value': metrics.get('initial_value', 0),
                'final_value': metrics.get('final_value', 0),
            }

            if 'training' in metrics and 'execution' in metrics['training']:
                episode_metrics['loss'] = episode_loss

            if 'baselines' in metrics:
                episode_metrics['baselines'] = metrics['baselines']

            monitor.log_episode(episode_metrics)

            # Update progress bar with metrics
            avg_return = sum(recent_returns) / len(recent_returns) if recent_returns else 0
            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0

            pbar.set_postfix({
                'ret': f'{avg_return:.4f}',
                'loss': f'{avg_loss:.4f}',
                'val': f'{best_val_return:.4f}'
            })

            # Validation
            if episode % validation_interval == 0 and episode > 0:
                pbar.write(f"\n🔍 Validation at episode {episode}...")

                with QuietLogger(logging.ERROR if not verbose else logging.INFO):
                    val_metrics = system.validate()

                val_return = val_metrics.get('rl_return', 0)
                monitor.log_validation(episode, val_return)

                pbar.write(f"   Return: {val_return:.4f}")
                pbar.write(f"   vs Baselines: {val_metrics.get('vs_baselines_score', 0):.2f}")

                # Save best model
                if val_return > best_val_return:
                    best_val_return = val_return
                    patience_counter = 0

                    checkpoint_mgr.save(
                        system.agent,
                        episode=episode,
                        metrics={'val_return': val_return},
                        monitor=monitor,
                        is_best=True
                    )
                    pbar.write(f"   ✅ New best model! Val return: {val_return:.4f}\n")
                else:
                    patience_counter += 1
                    pbar.write(f"   No improvement (patience: {patience_counter}/{patience})\n")

                if patience_counter >= patience:
                    pbar.write(f"\n⏹️  Early stopping at episode {episode}")
                    break

            # Checkpointing
            if episode % checkpoint_interval == 0 and episode > 0:
                with QuietLogger():
                    checkpoint_mgr.save(
                        system.agent,
                        episode=episode,
                        metrics=episode_metrics,
                        monitor=monitor
                    )
                pbar.write(f"💾 Checkpoint saved at episode {episode}")

            # Plotting (silent)
            if episode % plot_interval == 0 and episode > 0:
                with QuietLogger():
                    monitor.plot(save=True, show=False)

        pbar.close()

        # Final summary
        training_time = time.time() - training_start

        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Episodes: {num_episodes - start_episode}")
        print(f"Time: {training_time / 3600:.2f} hours")
        print(f"Avg time/episode: {training_time / (num_episodes - start_episode):.2f}s")
        print(f"Best validation return: {best_val_return:.4f}")
        print("=" * 70 + "\n")

        # Save final checkpoint
        with QuietLogger():
            checkpoint_mgr.save(
                system.agent,
                episode=num_episodes - 1,
                metrics={'training_time': training_time},
                monitor=monitor
            )
            monitor.plot(save=True, show=False)
            monitor.save()

        print("📊 Training summary saved to training_logs/")
        print("📈 Plots saved to training_logs/training_progress.png")

    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted by user")
        print("💾 Saving checkpoint...")

        with QuietLogger():
            checkpoint_mgr.save(
                system.agent,
                episode=episode,
                metrics={'interrupted': True},
                monitor=monitor
            )
            monitor.plot(save=True, show=False)
            monitor.save()

        print("✅ Progress saved - you can resume with:")
        print(f"   train_with_progress(resume_from='training_logs/checkpoints/checkpoint_ep{episode}.pth')")

    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")

        if verbose:
            import traceback
            traceback.print_exc()

        # Try to save what we have
        try:
            with QuietLogger():
                checkpoint_mgr.save(
                    system.agent,
                    episode=episode,
                    metrics={'error': str(e)},
                    monitor=monitor
                )
                monitor.save()
            print("✅ Emergency save completed")
        except:
            print("❌ Could not save checkpoint")


def continue_training(checkpoint_path: str = None, additional_episodes: int = 5000,
                      verbose: bool = False):
    """
    Quick function to continue training from a checkpoint

    Args:
        checkpoint_path: Optional specific checkpoint (if None, uses latest)
        additional_episodes: How many more episodes to train
        verbose: Enable detailed logging
    """
    from training_monitor import CheckpointManager
    import torch

    if checkpoint_path is None:
        # Auto-find latest
        checkpoint_mgr = CheckpointManager('training_logs/checkpoints')
        checkpoints = sorted(
            checkpoint_mgr.checkpoint_dir.glob('checkpoint_ep*.pth'),
            key=lambda p: int(p.stem.split('ep')[1])
        )

        if not checkpoints:
            print("❌ No checkpoints found to continue from!")
            return

        checkpoint_path = str(checkpoints[-1])

    # Load checkpoint to get episode number
    checkpoint = torch.load(checkpoint_path)
    current_episode = checkpoint['episode']

    print(f"\n📂 Resuming from: {checkpoint_path}")
    print(f"📊 Current episode: {current_episode}")
    print(f"➕ Additional episodes: {additional_episodes}")
    print(f"🎯 Target episode: {current_episode + additional_episodes}\n")

    train_with_progress(
        num_episodes=current_episode + additional_episodes,
        resume_from=checkpoint_path,
        auto_resume=False,
        verbose=verbose
    )


def quick_train(episodes: int = 10000, verbose: bool = False):
    """
    Simplest possible training interface

    Args:
        episodes: Total episodes to train
        verbose: Enable detailed logging
    """
    train_with_progress(
        num_episodes=episodes,
        checkpoint_interval=50,
        validation_interval=50,
        verbose=verbose
    )


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Train RL trading agent with progress tracking')
    parser.add_argument('--episodes', type=int, default=810,
                        help='Number of episodes to train (default: 10000)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from specific checkpoint')
    parser.add_argument('--continue', dest='continue_training', action='store_true',
                        help='Continue from latest checkpoint with additional episodes')
    parser.add_argument('--additional', type=int, default=5000,
                        help='Additional episodes when using --continue (default: 5000)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable detailed logging')
    parser.add_argument('--no-auto-resume', action='store_true',
                        help='Disable automatic checkpoint resumption')

    args = parser.parse_args()

    if args.continue_training:
        # Continue training mode
        continue_training(
            checkpoint_path=args.resume,
            additional_episodes=args.additional,
            verbose=args.verbose
        )
    else:
        # Normal training mode
        train_with_progress(
            num_episodes=args.episodes,
            resume_from=args.resume,
            auto_resume=False,
            verbose=args.verbose
        )