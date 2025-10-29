#!/bin/bash
# setup_cicd.sh - Quick setup for CI/CD and Docker deployment

set -e

echo "=============================================="
echo "Trading Bot CI/CD Setup"
echo "=============================================="
echo ""

# ============================================================================
# Check Prerequisites
# ============================================================================

echo "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Install from: https://docs.docker.com/get-docker/"
    exit 1
fi
echo "✓ Docker installed"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Install from: https://docs.docker.com/compose/install/"
    exit 1
fi
echo "✓ Docker Compose installed"

# Check Git
if ! command -v git &> /dev/null; then
    echo "❌ Git not found. Please install git first."
    exit 1
fi
echo "✓ Git installed"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi
echo "✓ Python installed"

echo ""

# ============================================================================
# Create Directory Structure
# ============================================================================

echo "Creating directory structure..."

mkdir -p .github/workflows
mkdir -p models
mkdir -p training_logs/checkpoints
mkdir -p simulation_cache
mkdir -p logs
mkdir -p tests

echo "✓ Directories created"
echo ""

# ============================================================================
# Create Configuration Files
# ============================================================================

echo "Creating configuration files..."

# Create .env.example
cat > .env.example << 'EOF'
# API Credentials
ALPACA_API_KEY=your_key_here
ALPACA_API_SECRET=your_secret_here

# Application Settings
VERSION=v1.0.0
MODE=paper_trading
LOG_LEVEL=INFO
ENVIRONMENT=development

# Trading Parameters
INITIAL_CAPITAL=100000
MAX_POSITIONS=20
DECISION_INTERVAL_MINUTES=5

# Training Parameters
TRAINING_EPISODES=10000
CHECKPOINT_INTERVAL=50

# Docker Settings
DOCKER_REGISTRY=ghcr.io
IMAGE_NAME=yourusername/trading-bot
EOF

echo "✓ Created .env.example"

# Create deployment_config.json
cat > deployment_config.json << 'EOF'
{
  "registry": "ghcr.io",
  "repository": "yourusername/trading-bot",
  "environments": {
    "development": {
      "url": "localhost:5000",
      "replicas": 1,
      "resources": {
        "cpu": "0.5",
        "memory": "512Mi"
      }
    },
    "staging": {
      "url": "staging.trading-bot.example.com",
      "replicas": 2,
      "resources": {
        "cpu": "1",
        "memory": "1Gi"
      }
    },
    "production": {
      "url": "trading-bot.example.com",
      "replicas": 3,
      "resources": {
        "cpu": "2",
        "memory": "2Gi"
      }
    }
  }
}
EOF

echo "✓ Created deployment_config.json"

# Create .dockerignore
cat > .dockerignore << 'EOF'
# Git
.git
.gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Testing
.pytest_cache/
.coverage
htmlcov/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs and data
logs/
*.log

# Training artifacts (use volumes instead)
training_logs/
models/*.pth
simulation_cache/

# Documentation
*.md
docs/

# CI/CD
.github/
EOF

echo "✓ Created .dockerignore"

# Update .gitignore
cat >> .gitignore << 'EOF'

# CI/CD
.env
versions.json
docker-compose.*.yml

# Deployment
deployment_config.json
EOF

echo "✓ Updated .gitignore"
echo ""

# ============================================================================
# Make Scripts Executable
# ============================================================================

echo "Making scripts executable..."

chmod +x version_manager.py 2>/dev/null || true
chmod +x docker-entrypoint.sh 2>/dev/null || true

echo "✓ Scripts are executable"
echo ""

# ============================================================================
# Install Python Dependencies
# ============================================================================

echo "Installing Python dependencies..."

pip install -q pyyaml 2>/dev/null || pip3 install -q pyyaml

echo "✓ Dependencies installed"
echo ""

# ============================================================================
# Initialize Git Repository (if needed)
# ============================================================================

if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit with CI/CD setup"
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already exists"
fi
echo ""

# ============================================================================
# Test Docker Setup
# ============================================================================

echo "Testing Docker setup..."

# Build test image
docker build --target runtime -t trading-bot:test . > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Docker build successful"
    docker rmi trading-bot:test > /dev/null 2>&1
else
    echo "⚠️  Docker build had issues (may need Dockerfile)"
fi
echo ""

# ============================================================================
# Create Example Test File
# ============================================================================

echo "Creating example test file..."

cat > tests/test_example.py << 'EOF'
"""Example test file for CI/CD pipeline"""
import pytest

def test_basic():
    """Basic test to ensure pytest works"""
    assert 1 + 1 == 2

def test_import():
    """Test that we can import the main modules"""
    try:
        import numpy as np
        import pandas as pd
        import torch
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
EOF

echo "✓ Created example test"
echo ""

# ============================================================================
# User Configuration
# ============================================================================

echo "=============================================="
echo "Configuration Required"
echo "=============================================="
echo ""

read -p "Enter your GitHub username: " github_username
read -p "Enter your repository name: " repo_name

# Update deployment_config.json
if command -v jq &> /dev/null; then
    jq --arg user "$github_username" --arg repo "$repo_name" \
       '.repository = $user + "/" + $repo' \
       deployment_config.json > deployment_config.json.tmp
    mv deployment_config.json.tmp deployment_config.json
    echo "✓ Updated deployment config"
else
    echo "⚠️  jq not found - please manually update deployment_config.json"
    echo "   Set repository to: $github_username/$repo_name"
fi

echo ""

# ============================================================================
# Create .env File
# ============================================================================

echo "Setting up environment variables..."
echo ""

if [ -f ".env" ]; then
    echo "⚠️  .env file already exists"
    read -p "Overwrite? (y/n): " overwrite
    if [ "$overwrite" != "y" ]; then
        echo "Keeping existing .env"
    else
        cp .env.example .env
        echo "✓ Created new .env file"
    fi
else
    cp .env.example .env
    echo "✓ Created .env file"
fi

echo ""
echo "⚠️  IMPORTANT: Edit .env and add your API keys:"
echo "   - ALPACA_API_KEY"
echo "   - ALPACA_API_SECRET"
echo ""

# ============================================================================
# Summary and Next Steps
# ============================================================================

echo "=============================================="
echo "Setup Complete! ✅"
echo "=============================================="
echo ""
echo "What's been set up:"
echo "  ✓ Directory structure"
echo "  ✓ Configuration files"
echo "  ✓ Git repository"
echo "  ✓ Example tests"
echo ""
echo "Next steps:"
echo ""
echo "1. Configure API credentials:"
echo "   nano .env"
echo ""
echo "2. Update deployment_config.json with your details"
echo ""
echo "3. Create your first version:"
echo "   python version_manager.py create v1.0.0 \"Initial release\""
echo ""
echo "4. Test locally with Docker:"
echo "   docker-compose up"
echo ""
echo "5. Set up GitHub Actions:"
echo "   - Push code to GitHub"
echo "   - Add secrets in Settings → Secrets → Actions"
echo "   - Push a version tag: git push origin v1.0.0"
echo ""
echo "6. Read the deployment guide:"
echo "   cat DEPLOYMENT_GUIDE.md"
echo ""
echo "=============================================="
echo ""

# ============================================================================
# Quick Start Options
# ============================================================================

echo "Quick start options:"
echo ""
echo "a) Test Docker setup now"
echo "b) Create first version"
echo "c) Skip and configure manually"
echo ""
read -p "Choose option (a/b/c): " option

case $option in
    a)
        echo ""
        echo "Testing Docker setup..."
        docker-compose config
        echo ""
        echo "✓ Docker configuration is valid"
        echo ""
        echo "To start the bot:"
        echo "  docker-compose up"
        ;;
    b)
        echo ""
        read -p "Enter version (e.g., v1.0.0): " version
        read -p "Enter description: " description

        if [ -f "version_manager.py" ]; then
            python version_manager.py create "$version" "$description"
        else
            echo "⚠️  version_manager.py not found"
            echo "   Create version manually with git tag"
        fi
        ;;
    c)
        echo ""
        echo "Manual configuration selected"
        ;;
    *)
        echo "Invalid option"
        ;;
esac

echo ""
echo "For detailed documentation, see DEPLOYMENT_GUIDE.md"
echo ""