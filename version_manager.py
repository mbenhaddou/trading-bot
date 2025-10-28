#!/usr/bin/env python3
"""
Version Management and Deployment Tool

Usage:
    python version_manager.py list                          # List all versions
    python version_manager.py create v1.2.3 "Bug fixes"     # Create new version
    python version_manager.py deploy v1.2.3 production      # Deploy version
    python version_manager.py rollback production v1.2.2    # Rollback to version
    python version_manager.py status                        # Show deployment status
    python version_manager.py compare v1.2.2 v1.2.3        # Compare versions
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import yaml


class VersionManager:
    """Manage versions and deployments of the trading bot"""

    def __init__(self, config_file: str = "deployment_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.versions_file = Path("versions.json")
        self.versions = self._load_versions()

    def _load_config(self) -> Dict:
        """Load deployment configuration"""
        if self.config_file.exists():
            with open(self.config_file) as f:
                return json.load(f)

        # Default configuration
        default_config = {
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

        # Save default config
        with open(self.config_file, 'w') as f:
            json.dump(default_config, f, indent=2)

        print(f"Created default config: {self.config_file}")
        print(f"Please update with your GitHub username and URLs")

        return default_config

    def _load_versions(self) -> Dict:
        """Load version history"""
        if self.versions_file.exists():
            with open(self.versions_file) as f:
                return json.load(f)
        return {"versions": [], "deployments": {}}

    def _save_versions(self):
        """Save version history"""
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2)

    def list_versions(self):
        """List all available versions"""
        print("\n" + "=" * 80)
        print("AVAILABLE VERSIONS")
        print("=" * 80)

        if not self.versions["versions"]:
            print("\nNo versions found. Create one with:")
            print("  python version_manager.py create v1.0.0 \"Initial release\"")
            return

        # Sort by date (newest first)
        sorted_versions = sorted(
            self.versions["versions"],
            key=lambda x: x["created_at"],
            reverse=True
        )

        for version in sorted_versions:
            print(f"\n📦 {version['tag']}")
            print(f"   Created: {version['created_at']}")
            print(f"   Description: {version['description']}")
            print(f"   Commit: {version['commit_sha'][:8]}")

            # Show which environments have this version
            deployed_to = []
            for env, deployment in self.versions["deployments"].items():
                if deployment.get("version") == version["tag"]:
                    deployed_to.append(env)

            if deployed_to:
                print(f"   Deployed to: {', '.join(deployed_to)}")

        print("\n" + "=" * 80 + "\n")

    def create_version(self, tag: str, description: str):
        """Create a new version tag"""
        print(f"\n🏷️  Creating version {tag}...")

        # Validate tag format
        if not tag.startswith('v'):
            print("❌ Error: Version tag must start with 'v' (e.g., v1.0.0)")
            return False

        # Check if tag already exists
        if any(v["tag"] == tag for v in self.versions["versions"]):
            print(f"❌ Error: Version {tag} already exists")
            return False

        # Get current git commit
        try:
            commit_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True
            ).strip()
        except subprocess.CalledProcessError:
            print("❌ Error: Failed to get git commit. Are you in a git repository?")
            print("\nInitialize git with:")
            print("  git init")
            print("  git add .")
            print("  git commit -m 'Initial commit'")
            return False

        # Create git tag
        try:
            subprocess.run(
                ["git", "tag", "-a", tag, "-m", description],
                check=True
            )
            print(f"✅ Git tag created: {tag}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating git tag: {e}")
            return False

        # Add to versions list
        version_info = {
            "tag": tag,
            "description": description,
            "commit_sha": commit_sha,
            "created_at": datetime.now().isoformat(),
            "created_by": os.getenv("USER", "unknown")
        }

        self.versions["versions"].append(version_info)
        self._save_versions()

        print(f"\n✅ Version {tag} created successfully")
        print(f"   Commit: {commit_sha[:8]}")
        print(f"\nNext steps:")
        print(f"1. Push tag to remote: git push origin {tag}")
        print(f"2. Build Docker image: docker build -t trading-bot:{tag} .")
        print(f"3. Deploy: python version_manager.py deploy {tag} <environment>")

        return True

    def build_image(self, tag: str):
        """Build Docker image for a version"""
        print(f"\n🔨 Building Docker image for {tag}...")

        # Get version info
        version_info = next(
            (v for v in self.versions["versions"] if v["tag"] == tag),
            None
        )

        if not version_info:
            print(f"❌ Error: Version {tag} not found")
            return False

        # Build Docker image
        registry = self.config["registry"]
        repo = self.config["repository"]
        image_name = f"{registry}/{repo}:{tag}"

        build_args = [
            "docker", "build",
            "--build-arg", f"VERSION={tag}",
            "--build-arg", f"BUILD_DATE={datetime.now().isoformat()}",
            "--build-arg", f"VCS_REF={version_info['commit_sha']}",
            "-t", image_name,
            "-t", f"{registry}/{repo}:latest",
            "."
        ]

        print(f"\nBuilding: {image_name}")
        try:
            subprocess.run(build_args, check=True)
            print(f"\n✅ Image built successfully: {image_name}")
            print(f"\nNext steps:")
            print(f"1. Test locally: docker run {image_name}")
            print(f"2. Push to registry: docker push {image_name}")
            print(f"3. Deploy: python version_manager.py deploy {tag} <environment>")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error building image: {e}")
            return False

        return True

    def deploy(self, tag: str, environment: str):
        """Deploy a version to an environment"""
        print(f"\n🚀 Deploying {tag} to {environment}...")

        # Validate environment
        if environment not in self.config["environments"]:
            print(f"❌ Error: Unknown environment '{environment}'")
            print(f"   Available: {', '.join(self.config['environments'].keys())}")
            return False

        # Validate version
        version_info = next(
            (v for v in self.versions["versions"] if v["tag"] == tag),
            None
        )

        if not version_info:
            print(f"❌ Error: Version {tag} not found")
            print(f"\nCreate version first:")
            print(f"  python version_manager.py create {tag} \"Description\"")
            return False

        # Get environment config
        env_config = self.config["environments"][environment]

        # Confirm deployment
        if environment == "production":
            print(f"\n⚠️  WARNING: Deploying to PRODUCTION")
            print(f"   Version: {tag}")
            print(f"   Replicas: {env_config['replicas']}")
            confirm = input("\nType 'yes' to confirm: ")
            if confirm.lower() != 'yes':
                print("Deployment cancelled")
                return False

        # Create docker-compose override for this deployment
        compose_override = self._create_compose_override(tag, environment, env_config)

        # Save override file
        override_file = Path(f"docker-compose.{environment}.yml")
        with open(override_file, 'w') as f:
            yaml.dump(compose_override, f)

        print(f"Created: {override_file}")

        # Deploy using docker-compose
        compose_files = [
            "-f", "docker-compose.yml",
            "-f", str(override_file)
        ]

        registry = self.config["registry"]
        repo = self.config["repository"]

        env_vars = {
            "VERSION": tag,
            "DOCKER_REGISTRY": registry,
            "IMAGE_NAME": repo
        }

        deploy_cmd = ["docker-compose"] + compose_files + ["up", "-d"]

        print(f"\nRunning: {' '.join(deploy_cmd)}")

        try:
            subprocess.run(
                deploy_cmd,
                check=True,
                env={**os.environ, **env_vars}
            )

            # Record deployment
            self.versions["deployments"][environment] = {
                "version": tag,
                "deployed_at": datetime.now().isoformat(),
                "deployed_by": os.getenv("USER", "unknown"),
                "config": env_config
            }
            self._save_versions()

            print(f"\n✅ Deployment successful!")
            print(f"   Environment: {environment}")
            print(f"   Version: {tag}")
            print(f"   URL: {env_config.get('url', 'N/A')}")
            print(f"\nView logs:")
            print(f"  docker-compose {' '.join(compose_files)} logs -f")

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Deployment failed: {e}")
            return False

    def _create_compose_override(self, tag: str, environment: str, config: Dict) -> Dict:
        """Create docker-compose override for deployment"""
        return {
            "version": "3.9",
            "services": {
                "trading-bot": {
                    "image": f"{self.config['registry']}/{self.config['repository']}:{tag}",
                    "deploy": {
                        "replicas": config["replicas"],
                        "resources": {
                            "limits": {
                                "cpus": config["resources"]["cpu"],
                                "memory": config["resources"]["memory"]
                            }
                        }
                    },
                    "environment": {
                        "ENVIRONMENT": environment,
                        "VERSION": tag
                    }
                }
            }
        }

    def rollback(self, environment: str, tag: str):
        """Rollback to a previous version"""
        print(f"\n⏮️  Rolling back {environment} to {tag}...")

        current = self.versions["deployments"].get(environment, {}).get("version")

        if current == tag:
            print(f"❌ Error: {environment} is already running {tag}")
            return False

        print(f"   Current version: {current}")
        print(f"   Target version: {tag}")
        confirm = input("\nConfirm rollback? (yes/no): ")

        if confirm.lower() != 'yes':
            print("Rollback cancelled")
            return False

        # Deploy the target version
        return self.deploy(tag, environment)

    def status(self):
        """Show deployment status across all environments"""
        print("\n" + "=" * 80)
        print("DEPLOYMENT STATUS")
        print("=" * 80)

        for env_name, env_config in self.config["environments"].items():
            deployment = self.versions["deployments"].get(env_name, {})

            print(f"\n🌍 {env_name.upper()}")
            print(f"   URL: {env_config.get('url', 'N/A')}")

            if deployment:
                print(f"   Version: {deployment['version']}")
                print(f"   Deployed at: {deployment['deployed_at']}")
                print(f"   Deployed by: {deployment['deployed_by']}")
                print(f"   Replicas: {deployment['config']['replicas']}")
            else:
                print(f"   Status: Not deployed")

        print("\n" + "=" * 80 + "\n")

    def compare(self, tag1: str, tag2: str):
        """Compare two versions"""
        print(f"\n📊 Comparing {tag1} vs {tag2}...")

        try:
            # Get commit range
            result = subprocess.run(
                ["git", "log", "--oneline", f"{tag1}..{tag2}"],
                capture_output=True,
                text=True,
                check=True
            )

            commits = result.stdout.strip().split('\n') if result.stdout.strip() else []

            print(f"\nChanges from {tag1} to {tag2}:")
            print(f"   Total commits: {len(commits)}")

            if commits:
                print("\nCommit history:")
                for commit in commits[:20]:  # Show first 20
                    print(f"   - {commit}")

                if len(commits) > 20:
                    print(f"   ... and {len(commits) - 20} more")
            else:
                print("   No commits between these versions")

        except subprocess.CalledProcessError as e:
            print(f"❌ Error comparing versions: {e}")
            print("\nMake sure both tags exist:")
            print(f"  git tag -l")


def main():
    parser = argparse.ArgumentParser(
        description="Version Management and Deployment Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python version_manager.py list
  python version_manager.py create v1.2.3 "Bug fixes and improvements"
  python version_manager.py build v1.2.3
  python version_manager.py deploy v1.2.3 production
  python version_manager.py rollback production v1.2.2
  python version_manager.py status
  python version_manager.py compare v1.2.2 v1.2.3
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # List command
    subparsers.add_parser('list', help='List all versions')

    # Create command
    create_parser = subparsers.add_parser('create', help='Create new version')
    create_parser.add_argument('tag', help='Version tag (e.g., v1.2.3)')
    create_parser.add_argument('description', help='Version description')

    # Build command
    build_parser = subparsers.add_parser('build', help='Build Docker image')
    build_parser.add_argument('tag', help='Version tag')

    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy version')
    deploy_parser.add_argument('tag', help='Version tag')
    deploy_parser.add_argument('environment', help='Target environment')

    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback to version')
    rollback_parser.add_argument('environment', help='Target environment')
    rollback_parser.add_argument('tag', help='Version tag to rollback to')

    # Status command
    subparsers.add_parser('status', help='Show deployment status')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two versions')
    compare_parser.add_argument('tag1', help='First version tag')
    compare_parser.add_argument('tag2', help='Second version tag')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Create version manager instance
    try:
        manager = VersionManager()
    except Exception as e:
        print(f"❌ Error initializing version manager: {e}")
        return

    # Execute command
    if args.command == 'list':
        manager.list_versions()
    elif args.command == 'create':
        manager.create_version(args.tag, args.description)
    elif args.command == 'build':
        manager.build_image(args.tag)
    elif args.command == 'deploy':
        manager.deploy(args.tag, args.environment)
    elif args.command == 'rollback':
        manager.rollback(args.environment, args.tag)
    elif args.command == 'status':
        manager.status()
    elif args.command == 'compare':
        manager.compare(args.tag1, args.tag2)


if __name__ == "__main__":
    main()