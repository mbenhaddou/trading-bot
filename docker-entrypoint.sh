#!/bin/bash
# docker-entrypoint.sh
# Entrypoint script for trading bot container

set -e

# ============================================================================
# Environment Setup
# ============================================================================

echo "=============================================="
echo "RL Trading Bot - Version ${VERSION:-unknown}"
echo "=============================================="
echo "Mode: ${MODE:-paper_trading}"
echo "Environment: ${ENVIRONMENT:-development}"
echo ""

# Display build information if available
if [ -f /app/BUILD_INFO ]; then
    cat /app/BUILD_INFO
    echo ""
fi

# ============================================================================
# Pre-flight Checks
# ============================================================================

echo "Running pre-flight checks..."

# Check for required environment variables
if [ "$MODE" = "live_trading" ] || [ "$MODE" = "paper_trading" ]; then
    if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_API_SECRET" ]; then
        echo "ERROR: ALPACA_API_KEY and ALPACA_API_SECRET must be set for trading mode"
        exit 1
    fi
    echo "✓ Alpaca credentials found"
fi

# Check for required files
if [ ! -f "/app/config.json" ]; then
    echo "ERROR: config.json not found"
    exit 1
fi
echo "✓ Configuration file found"

# Create necessary directories
mkdir -p /app/models /app/logs /app/training_logs /app/simulation_cache
echo "✓ Directories created"

# ============================================================================
# Mode-Specific Setup
# ============================================================================

case "$MODE" in
    training)
        echo ""
        echo "Starting TRAINING mode..."
        echo "Episodes: ${TRAINING_EPISODES:-10000}"
        echo "Checkpoint interval: ${CHECKPOINT_INTERVAL:-50}"

        # Check if resuming from checkpoint
        if [ -d "/app/training_logs/checkpoints" ]; then
            CHECKPOINT_COUNT=$(ls -1 /app/training_logs/checkpoints/*.pth 2>/dev/null | wc -l)
            if [ "$CHECKPOINT_COUNT" -gt 0 ]; then
                echo "Found $CHECKPOINT_COUNT existing checkpoint(s)"
                echo "Training will auto-resume from latest checkpoint"
            fi
        fi
        ;;

    paper_trading)
        echo ""
        echo "Starting PAPER TRADING mode..."
        echo "Decision interval: ${DECISION_INTERVAL_MINUTES:-5} minutes"

        # Check for trained model
        if [ -f "/app/models/best_model.pth" ]; then
            echo "✓ Found trained model: best_model.pth"
        else
            echo "⚠️  No trained model found - will need to train first"
        fi
        ;;

    live_trading)
        echo ""
        echo "⚠️  Starting LIVE TRADING mode..."
        echo "⚠️  This will trade with REAL money!"
        echo ""

        # Extra confirmation for live trading
        if [ "$ENVIRONMENT" = "production" ]; then
            echo "Waiting 10 seconds before starting (press Ctrl+C to cancel)..."
            sleep 10
        fi

        # Check for trained model
        if [ ! -f "/app/models/best_model.pth" ]; then
            echo "ERROR: No trained model found for live trading"
            exit 1
        fi
        echo "✓ Using model: best_model.pth"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes: training, paper_trading, live_trading"
        exit 1
        ;;
esac

# ============================================================================
# Health Monitoring Setup
# ============================================================================

# Start background health monitor
(
    while true; do
        sleep 60

        # Check if main process is still running
        if ! pgrep -f "python" > /dev/null; then
            echo "WARNING: Main process not running"
        fi

        # Check disk space
        DISK_USAGE=$(df -h /app | tail -1 | awk '{print $5}' | sed 's/%//')
        if [ "$DISK_USAGE" -gt 90 ]; then
            echo "WARNING: Disk usage at ${DISK_USAGE}%"
        fi

    done
) &

# ============================================================================
# Signal Handling
# ============================================================================

# Graceful shutdown handler
shutdown() {
    echo ""
    echo "Received shutdown signal, cleaning up..."

    # Save any in-progress work
    if [ -f "/tmp/trading_session.pid" ]; then
        MAIN_PID=$(cat /tmp/trading_session.pid)
        echo "Stopping main process (PID: $MAIN_PID)..."
        kill -SIGTERM $MAIN_PID 2>/dev/null || true
        wait $MAIN_PID 2>/dev/null || true
    fi

    # Cleanup
    echo "Cleanup complete"
    exit 0
}

trap shutdown SIGTERM SIGINT

# ============================================================================
# Start Application
# ============================================================================

echo ""
echo "Starting application..."
echo "=============================================="
echo ""

# Save PID for signal handling
echo $$ > /tmp/trading_session.pid

# Execute the command passed to the container
exec "$@"