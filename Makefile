# Makefile for Trading Bot CI/CD Operations
# Usage: make <target>

.PHONY: help setup build test deploy clean

# Default target
.DEFAULT_GOAL := help

# Load environment variables
-include .env
export

# Variables
VERSION ?= latest
ENVIRONMENT ?= development
DOCKER_REGISTRY ?= ghcr.io
IMAGE_NAME ?= yourusername/trading-bot
FULL_IMAGE := $(DOCKER_REGISTRY)/$(IMAGE_NAME):$(VERSION)

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

##@ Help

help: ## Display this help message
	@echo ""
	@echo "$(BLUE)Trading Bot CI/CD Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(CYAN)<target>$(NC)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(CYAN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""

##@ Setup & Installation

setup: ## Run initial setup
	@echo "$(GREEN)Running setup...$(NC)"
	@bash setup_cicd.sh

install-deps: ## Install Python dependencies
	@echo "$(GREEN)Installing dependencies...$(NC)"
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	@echo "$(GREEN)Installing dev dependencies...$(NC)"
	pip install -r requirements.txt pytest pytest-cov black flake8 mypy

init-git: ## Initialize git repository
	@if [ ! -d ".git" ]; then \
		git init; \
		git add .; \
		git commit -m "Initial commit"; \
		echo "$(GREEN)✓ Git initialized$(NC)"; \
	else \
		echo "$(YELLOW)Git repository already exists$(NC)"; \
	fi

##@ Version Management

version-list: ## List all versions
	@python version_manager.py list

version-create: ## Create new version (make version-create VERSION=v1.2.3 DESC="Description")
	@if [ -z "$(VERSION)" ] || [ -z "$(DESC)" ]; then \
		echo "$(RED)Error: VERSION and DESC required$(NC)"; \
		echo "Usage: make version-create VERSION=v1.2.3 DESC=\"Bug fixes\""; \
		exit 1; \
	fi
	@python version_manager.py create $(VERSION) "$(DESC)"

version-compare: ## Compare two versions (make version-compare V1=v1.2.2 V2=v1.2.3)
	@if [ -z "$(V1)" ] || [ -z "$(V2)" ]; then \
		echo "$(RED)Error: V1 and V2 required$(NC)"; \
		exit 1; \
	fi
	@python version_manager.py compare $(V1) $(V2)

##@ Docker Operations

docker-build: ## Build Docker image (make docker-build VERSION=v1.2.3)
	@echo "$(GREEN)Building Docker image: $(VERSION)$(NC)"
	docker build \
		--build-arg VERSION=$(VERSION) \
		--build-arg BUILD_DATE=$(shell date -u +"%Y-%m-%dT%H:%M:%SZ") \
		--build-arg VCS_REF=$(shell git rev-parse HEAD) \
		-t $(FULL_IMAGE) \
		-t $(DOCKER_REGISTRY)/$(IMAGE_NAME):latest \
		.
	@echo "$(GREEN)✓ Build complete: $(FULL_IMAGE)$(NC)"

docker-build-training: ## Build training image
	@echo "$(GREEN)Building training image$(NC)"
	docker build --target training -t $(FULL_IMAGE)-training .

docker-build-dev: ## Build development image
	@echo "$(GREEN)Building development image$(NC)"
	docker build --target development -t $(FULL_IMAGE)-dev .

docker-push: ## Push image to registry
	@echo "$(GREEN)Pushing $(FULL_IMAGE)$(NC)"
	docker push $(FULL_IMAGE)
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME):latest

docker-pull: ## Pull image from registry
	@echo "$(GREEN)Pulling $(FULL_IMAGE)$(NC)"
	docker pull $(FULL_IMAGE)

docker-clean: ## Remove local Docker images
	@echo "$(YELLOW)Cleaning Docker images...$(NC)"
	docker rmi $(FULL_IMAGE) 2>/dev/null || true
	docker rmi $(DOCKER_REGISTRY)/$(IMAGE_NAME):latest 2>/dev/null || true
	docker image prune -f

##@ Development

dev: ## Start development environment
	@echo "$(GREEN)Starting development environment$(NC)"
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

dev-build: ## Build and start development environment
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build

shell: ## Open shell in development container
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml run --rm trading-bot /bin/bash

##@ Testing

test: ## Run tests
	@echo "$(GREEN)Running tests...$(NC)"
	pytest tests/ -v

test-coverage: ## Run tests with coverage
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	pytest tests/ -v --cov=autonomous_trading_bot --cov-report=html --cov-report=term

test-docker: ## Run tests in Docker
	docker-compose run --rm trading-bot pytest tests/ -v

lint: ## Run code linting
	@echo "$(GREEN)Running linters...$(NC)"
	black --check autonomous_trading_bot/
	flake8 autonomous_trading_bot/

format: ## Format code
	@echo "$(GREEN)Formatting code...$(NC)"
	black autonomous_trading_bot/

##@ Deployment

deploy-dev: ## Deploy to development
	@echo "$(GREEN)Deploying $(VERSION) to development$(NC)"
	VERSION=$(VERSION) docker-compose up -d

deploy-staging: ## Deploy to staging (make deploy-staging VERSION=v1.2.3)
	@echo "$(YELLOW)Deploying $(VERSION) to staging$(NC)"
	@python version_manager.py deploy $(VERSION) staging

deploy-prod: ## Deploy to production (make deploy-prod VERSION=v1.2.3)
	@echo "$(RED)Deploying $(VERSION) to production$(NC)"
	@echo "$(YELLOW)WARNING: This will deploy to PRODUCTION$(NC)"
	@read -p "Type 'yes' to confirm: " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		python version_manager.py deploy $(VERSION) production; \
	else \
		echo "Deployment cancelled"; \
	fi

status: ## Show deployment status
	@python version_manager.py status

rollback: ## Rollback environment (make rollback ENV=production VERSION=v1.2.2)
	@if [ -z "$(ENV)" ] || [ -z "$(VERSION)" ]; then \
		echo "$(RED)Error: ENV and VERSION required$(NC)"; \
		echo "Usage: make rollback ENV=production VERSION=v1.2.2"; \
		exit 1; \
	fi
	@python version_manager.py rollback $(ENV) $(VERSION)

##@ Running Modes

train: ## Start training
	@echo "$(GREEN)Starting training...$(NC)"
	docker-compose up trainer

train-interactive: ## Start training with live logs
	docker-compose up trainer --no-detach

paper-trade: ## Start paper trading
	@echo "$(GREEN)Starting paper trading...$(NC)"
	docker-compose up trading-bot

live-trade: ## Start live trading (CAUTION!)
	@echo "$(RED)WARNING: Starting LIVE trading with REAL money!$(NC)"
	@read -p "Type 'yes' to confirm: " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		MODE=live_trading docker-compose up trading-bot; \
	else \
		echo "Live trading cancelled"; \
	fi

##@ Monitoring & Logs

logs: ## Show logs (make logs SERVICE=trading-bot)
	@if [ -z "$(SERVICE)" ]; then \
		docker-compose logs -f; \
	else \
		docker-compose logs -f $(SERVICE); \
	fi

logs-tail: ## Tail logs (make logs-tail SERVICE=trading-bot LINES=100)
	@docker-compose logs -f --tail=$(LINES) $(SERVICE)

ps: ## Show running containers
	@docker-compose ps

stats: ## Show container stats
	@docker stats

health: ## Check container health
	@docker inspect --format='{{.State.Health.Status}}' trading-bot-paper || echo "Container not running"

##@ Maintenance

start: ## Start all services
	docker-compose up -d

stop: ## Stop all services
	docker-compose down

restart: ## Restart all services
	docker-compose restart

clean: ## Clean up containers, volumes, and cache
	@echo "$(YELLOW)Cleaning up...$(NC)"
	docker-compose down -v
	rm -rf __pycache__ .pytest_cache .coverage htmlcov/
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-all: ## Deep clean (removes models, logs, cache)
	@echo "$(RED)WARNING: This will remove all data including models!$(NC)"
	@read -p "Type 'yes' to confirm: " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		make clean; \
		rm -rf models/* training_logs/* simulation_cache/* logs/*; \
		echo "$(GREEN)✓ Deep clean complete$(NC)"; \
	else \
		echo "Clean cancelled"; \
	fi

backup: ## Backup models and training logs
	@echo "$(GREEN)Creating backup...$(NC)"
	@BACKUP_FILE="backup-$(shell date +%Y%m%d-%H%M%S).tar.gz"; \
	tar -czf $$BACKUP_FILE models/ training_logs/ config.json; \
	echo "$(GREEN)✓ Backup created: $$BACKUP_FILE$(NC)"

restore: ## Restore from backup (make restore BACKUP=backup-20250127-103000.tar.gz)
	@if [ -z "$(BACKUP)" ]; then \
		echo "$(RED)Error: BACKUP file required$(NC)"; \
		echo "Usage: make restore BACKUP=backup-20250127-103000.tar.gz"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Restoring from $(BACKUP)...$(NC)"
	tar -xzf $(BACKUP)
	@echo "$(GREEN)✓ Restore complete$(NC)"

##@ Quick Actions

quick-start: ## Quick start (build, train small model, paper trade)
	@echo "$(GREEN)Quick start sequence...$(NC)"
	@make docker-build VERSION=dev
	@echo "\n$(BLUE)Training small model...$(NC)"
	@TRAINING_EPISODES=100 docker-compose up trainer --abort-on-container-exit
	@echo "\n$(BLUE)Starting paper trading...$(NC)"
	@docker-compose up trading-bot

demo: ## Run demo mode (synthetic data, no API keys needed)
	@echo "$(GREEN)Starting demo mode...$(NC)"
	MODE=simulation docker-compose up

##@ CI/CD

ci-test: ## Run CI tests locally
	@echo "$(GREEN)Running CI test suite...$(NC)"
	pytest tests/ -v --cov=autonomous_trading_bot --cov-report=xml

ci-build: ## Build for CI
	docker build --target runtime -t test-image .

ci-validate: ## Validate all configs
	@echo "$(GREEN)Validating configurations...$(NC)"
	@python -c "import json; json.load(open('config.json'))" && echo "✓ config.json valid"
	@python -c "import json; json.load(open('deployment_config.json'))" && echo "✓ deployment_config.json valid"
	@docker-compose config >/dev/null && echo "✓ docker-compose.yml valid"

##@ Information

info: ## Show system information
	@echo ""
	@echo "$(BLUE)System Information$(NC)"
	@echo "=================="
	@echo "Version:       $(VERSION)"
	@echo "Environment:   $(ENVIRONMENT)"
	@echo "Registry:      $(DOCKER_REGISTRY)"
	@echo "Image:         $(IMAGE_NAME)"
	@echo "Full Image:    $(FULL_IMAGE)"
	@echo ""
	@echo "$(BLUE)Docker Info$(NC)"
	@echo "==========="
	@docker --version
	@docker-compose --version
	@echo ""
	@echo "$(BLUE)Git Info$(NC)"
	@echo "========"
	@git --version 2>/dev/null || echo "Git not installed"
	@git branch --show-current 2>/dev/null || echo "Not in git repository"
	@git describe --tags 2>/dev/null || echo "No tags found"
	@echo ""

env: ## Show environment variables
	@echo "$(BLUE)Environment Variables$(NC)"
	@echo "===================="
	@env | grep -E "VERSION|MODE|ENVIRONMENT|ALPACA" | sort