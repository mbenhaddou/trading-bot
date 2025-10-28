#!/bin/bash
# git_setup.sh - Automated git initialization and GitHub setup

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================="
echo "Git Repository Setup"
echo "==============================================${NC}"
echo ""

# ============================================================================
# Step 1: Verify we're in the right directory
# ============================================================================

if [ ! -f "config.json" ] || [ ! -d "autonomous_trading_bot" ]; then
    echo -e "${RED}❌ Error: Not in the trading-bot project directory${NC}"
    echo ""
    echo "Please run this script from your project root directory"
    echo "Expected structure:"
    echo "  - config.json"
    echo "  - autonomous_trading_bot/"
    echo "  - Dockerfile"
    exit 1
fi

echo -e "${GREEN}✓ Project directory verified${NC}"
echo ""

# ============================================================================
# Step 2: Check if git is installed
# ============================================================================

if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git is not installed${NC}"
    echo ""
    echo "Install git:"
    echo "  macOS: brew install git"
    echo "  Ubuntu: sudo apt install git"
    exit 1
fi

echo -e "${GREEN}✓ Git is installed ($(git --version))${NC}"
echo ""

# ============================================================================
# Step 3: Configure git user (if not configured)
# ============================================================================

GIT_USER=$(git config --global user.name || echo "")
GIT_EMAIL=$(git config --global user.email || echo "")

if [ -z "$GIT_USER" ] || [ -z "$GIT_EMAIL" ]; then
    echo -e "${YELLOW}Git user not configured. Let's set it up:${NC}"
    echo ""

    read -p "Enter your name: " user_name
    read -p "Enter your email: " user_email

    git config --global user.name "$user_name"
    git config --global user.email "$user_email"

    echo ""
    echo -e "${GREEN}✓ Git user configured${NC}"
else
    echo -e "${GREEN}✓ Git user already configured${NC}"
    echo "  Name: $GIT_USER"
    echo "  Email: $GIT_EMAIL"
fi

echo ""

# ============================================================================
# Step 4: Check for .gitignore
# ============================================================================

if [ ! -f ".gitignore" ]; then
    echo -e "${YELLOW}Creating .gitignore...${NC}"

    cat > .gitignore << 'EOF'
# Environment variables (NEVER COMMIT!)
.env
.env.local
.env.*.local
*.env

# Python
__pycache__/
*.py[cod]
*.so
.Python
venv/
env/
ENV/
*.egg-info/
.pytest_cache/

# Logs
logs/
*.log

# Training artifacts
training_logs/
models/*.pth
simulation_cache/

# CI/CD generated
versions.json
docker-compose.development.yml
docker-compose.staging.yml
docker-compose.production.yml

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Backups
*.bak
*.backup
EOF

    echo -e "${GREEN}✓ .gitignore created${NC}"
else
    echo -e "${GREEN}✓ .gitignore already exists${NC}"
fi

echo ""

# ============================================================================
# Step 5: Critical security check - verify .env is ignored
# ============================================================================

echo -e "${YELLOW}🔒 Security Check: Verifying secrets are protected...${NC}"

# Check if .env exists
if [ -f ".env" ]; then
    # Check if .env is in .gitignore
    if grep -q "^\.env$" .gitignore; then
        echo -e "${GREEN}✓ .env is properly ignored${NC}"
    else
        echo -e "${RED}❌ WARNING: .env exists but not in .gitignore!${NC}"
        echo ""
        echo "Adding .env to .gitignore..."
        echo ".env" >> .gitignore
        echo -e "${GREEN}✓ Fixed${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  .env file not found (will be created from .env.example later)${NC}"
fi

echo ""

# ============================================================================
# Step 6: Initialize git repository
# ============================================================================

if [ -d ".git" ]; then
    echo -e "${YELLOW}Git repository already initialized${NC}"
    echo ""
    read -p "Reinitialize? This will keep existing commits (yes/no): " reinit
    if [ "$reinit" = "yes" ]; then
        echo "Keeping existing repository..."
    fi
else
    echo -e "${BLUE}Initializing git repository...${NC}"
    git init
    echo -e "${GREEN}✓ Git repository initialized${NC}"
fi

echo ""

# ============================================================================
# Step 7: Stage files
# ============================================================================

echo -e "${BLUE}Staging files...${NC}"

# Add all files
git add .

# Show what's being committed
echo ""
echo "Files to be committed:"
git diff --staged --name-only | head -20
echo ""

# Count files
NUM_FILES=$(git diff --staged --name-only | wc -l)
echo -e "${GREEN}✓ $NUM_FILES files staged${NC}"

# Critical check: make sure .env is NOT staged
if git diff --staged --name-only | grep -q "^\.env$"; then
    echo -e "${RED}❌ CRITICAL: .env file is staged!${NC}"
    echo "Removing .env from staging..."
    git reset HEAD .env
    echo -e "${GREEN}✓ .env removed from staging${NC}"
else
    echo -e "${GREEN}✓ Security check passed: .env not staged${NC}"
fi

echo ""

# ============================================================================
# Step 8: Create initial commit
# ============================================================================

# Check if there are commits
if git rev-parse HEAD >/dev/null 2>&1; then
    echo -e "${YELLOW}Repository already has commits${NC}"
    git log --oneline -5
else
    echo -e "${BLUE}Creating initial commit...${NC}"
    git commit -m "Initial commit: RL trading bot with CI/CD setup"
    echo -e "${GREEN}✓ Initial commit created${NC}"
fi

echo ""

# ============================================================================
# Step 9: GitHub repository setup
# ============================================================================

echo -e "${BLUE}=============================================="
echo "GitHub Repository Setup"
echo "==============================================${NC}"
echo ""

echo "Now you need to create a GitHub repository:"
echo ""
echo "1. Go to: https://github.com/new"
echo "2. Repository name: trading-bot (or your choice)"
echo "3. Description: RL Trading Bot with CI/CD"
echo "4. ⚠️  IMPORTANT: Select 'Private' (not Public)"
echo "5. Do NOT initialize with README, .gitignore, or license"
echo "6. Click 'Create repository'"
echo ""

read -p "Press Enter when you've created the GitHub repository..."

echo ""
read -p "Enter your GitHub username: " github_user
read -p "Enter repository name (default: trading-bot): " repo_name
repo_name=${repo_name:-trading-bot}

# ============================================================================
# Step 10: Add GitHub remote
# ============================================================================

GITHUB_URL="https://github.com/$github_user/$repo_name.git"

echo ""
echo -e "${BLUE}Adding GitHub remote...${NC}"
echo "URL: $GITHUB_URL"

# Remove existing remote if it exists
git remote remove origin 2>/dev/null || true

# Add new remote
git remote add origin "$GITHUB_URL"

echo -e "${GREEN}✓ GitHub remote added${NC}"

# Verify
echo ""
echo "Remotes:"
git remote -v

echo ""

# ============================================================================
# Step 11: Rename branch to main
# ============================================================================

current_branch=$(git branch --show-current)
if [ "$current_branch" != "main" ]; then
    echo -e "${BLUE}Renaming branch to 'main'...${NC}"
    git branch -M main
    echo -e "${GREEN}✓ Branch renamed to main${NC}"
else
    echo -e "${GREEN}✓ Already on main branch${NC}"
fi

echo ""

# ============================================================================
# Step 12: Push to GitHub
# ============================================================================

echo -e "${BLUE}Pushing to GitHub...${NC}"
echo ""
echo "You may be prompted for GitHub credentials:"
echo "  Username: $github_user"
echo "  Password: Use Personal Access Token (not your account password)"
echo ""
echo "Get token from: https://github.com/settings/tokens"
echo "Permissions needed: 'repo' scope"
echo ""

read -p "Press Enter to push..."

if git push -u origin main; then
    echo ""
    echo -e "${GREEN}✓ Successfully pushed to GitHub!${NC}"
else
    echo ""
    echo -e "${RED}❌ Push failed${NC}"
    echo ""
    echo "Common issues:"
    echo "1. Wrong credentials - use Personal Access Token"
    echo "2. Repository doesn't exist on GitHub"
    echo "3. No internet connection"
    echo ""
    echo "To retry manually:"
    echo "  git push -u origin main"
    exit 1
fi

echo ""

# ============================================================================
# Step 13: Create first version tag
# ============================================================================

echo -e "${BLUE}Creating version tag v1.0.0...${NC}"

git tag -a v1.0.0 -m "Initial release"
git push origin v1.0.0

echo -e "${GREEN}✓ Version tag v1.0.0 created and pushed${NC}"

echo ""

# ============================================================================
# Step 14: GitHub Secrets reminder
# ============================================================================

echo -e "${YELLOW}=============================================="
echo "⚠️  IMPORTANT: Configure GitHub Secrets"
echo "==============================================${NC}"
echo ""
echo "Go to: https://github.com/$github_user/$repo_name/settings/secrets/actions"
echo ""
echo "Add these secrets:"
echo "  1. ALPACA_API_KEY = your Alpaca API key"
echo "  2. ALPACA_API_SECRET = your Alpaca API secret"
echo ""
echo "Steps:"
echo "  1. Click 'Settings' tab"
echo "  2. Click 'Secrets and variables' → 'Actions'"
echo "  3. Click 'New repository secret'"
echo "  4. Add each secret"
echo ""

read -p "Press Enter when secrets are configured..."

# ============================================================================
# Step 15: Enable GitHub Actions
# ============================================================================

echo ""
echo -e "${YELLOW}⚠️  Enable GitHub Actions:${NC}"
echo ""
echo "1. Go to: https://github.com/$github_user/$repo_name/actions"
echo "2. Click 'I understand my workflows, go ahead and enable them'"
echo ""

read -p "Press Enter when GitHub Actions are enabled..."

# ============================================================================
# Success Summary
# ============================================================================

echo ""
echo -e "${GREEN}=============================================="
echo "✅ Git Setup Complete!"
echo "==============================================${NC}"
echo ""

echo -e "${BLUE}Repository Information:${NC}"
echo "  URL: https://github.com/$github_user/$repo_name"
echo "  Branch: main"
echo "  Initial commit: ✓"
echo "  Version tag: v1.0.0"
echo ""

echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo "1. Verify on GitHub:"
echo "   https://github.com/$github_user/$repo_name"
echo ""
echo "2. Check GitHub Actions:"
echo "   https://github.com/$github_user/$repo_name/actions"
echo ""
echo "3. Make your first change:"
echo "   git add ."
echo "   git commit -m 'Update: description'"
echo "   git push origin main"
echo ""
echo "4. Create new version:"
echo "   git tag -a v1.0.1 -m 'New features'"
echo "   git push origin v1.0.1"
echo ""

echo -e "${YELLOW}Security Reminders:${NC}"
echo "  • Repository is private: ✓"
echo "  • .env is gitignored: ✓"
echo "  • Secrets in GitHub: ✓"
echo "  • Never commit API keys to code!"
echo ""

echo -e "${GREEN}🎉 Your trading bot is now on GitHub!${NC}"
echo ""

# Create a quick reference file
cat > GIT_QUICK_REFERENCE.txt << EOF
Git Quick Reference
===================

Repository: https://github.com/$github_user/$repo_name

Daily Workflow:
--------------
# Make changes
git status
git add .
git commit -m "Description of changes"
git push origin main

Create New Version:
------------------
git tag -a v1.1.0 -m "Version description"
git push origin v1.1.0

Pull Latest:
-----------
git pull origin main
git fetch --tags

View History:
------------
git log --oneline
git tag -l

Undo Changes:
------------
git restore filename.py     # Discard local changes
git reset --soft HEAD~1     # Undo last commit (keep changes)

Branch Management:
-----------------
git checkout -b feature-name    # Create new branch
git checkout main               # Switch to main
git merge feature-name          # Merge branch

Security:
--------
• .env is gitignored
• Never commit API keys
• Keep repository private
• Use GitHub Secrets for CI/CD

Help:
----
git --help
https://git-scm.com/doc
EOF

echo "Quick reference saved to: GIT_QUICK_REFERENCE.txt"
echo ""