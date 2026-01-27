#!/usr/bin/env python3
"""
Repository Analyzer for Phoenix Protocol

Automatically analyzes 10+ SOTA repositories to extract best practices:
- Testing patterns and frameworks
- CI/CD configurations
- Code organization strategies
- Documentation standards
- Performance optimization techniques

Usage:
    python scripts/repo_analyzer.py --output analysis_report.json --summary

Author: Phoenix Protocol Team
Date: January 2026
"""

import os
import json
import argparse
import requests
import time
from typing import Dict, List, Any
from pathlib import Path


class RepositoryAnalyzer:
    """Analyze multiple repositories to extract best practices."""

    def __init__(self, github_token: str = None):
        self.github_token = github_token
        self.session = requests.Session()
        if github_token:
            self.session.headers.update({"Authorization": f"token {github_token}"})

    def search_repositories(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search GitHub repositories based on query."""
        url = "https://api.github.com/search/repositories"
        params = {"q": query, "sort": "stars", "order": "desc", "per_page": max_results}

        repositories = []
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()

            for repo in response.json().get("items", []):
                repo_info = {
                    "name": repo["full_name"],
                    "clone_url": repo["clone_url"],
                    "stars": repo["stargazers_count"],
                    "language": repo.get("language", "Unknown"),
                    "description": repo.get("description", ""),
                    "topics": repo.get("topics", []),
                }
                repositories.append(repo_info)

        except Exception as e:
            print(f"Error searching repositories: {e}")

        return repositories

    def analyze_repository(self, repo_path: str) -> Dict[str, Any]:
        """Analyze a single repository for patterns."""
        analysis = {
            "path": repo_path,
            "structure": self._analyze_structure(repo_path),
            "testing": self._analyze_testing(repo_path),
            "cicd": self._analyze_cicd(repo_path),
            "documentation": self._analyze_documentation(repo_path),
            "performance": self._analyze_performance(repo_path),
            "quality": self._analyze_code_quality(repo_path),
        }
        return analysis

    def _analyze_structure(self, repo_path: str) -> Dict:
        """Analyze repository structure."""
        structure = {}

        # Check for common directories
        common_dirs = ["src", "tests", "docs", "examples", "scripts", "tools"]
        structure["directories"] = {}

        for dir_name in common_dirs:
            dir_path = os.path.join(repo_path, dir_name)
            structure["directories"][dir_name] = os.path.exists(dir_path)

        # Check for configuration files
        config_files = [
            "README.md",
            "setup.py",
            "pyproject.toml",
            "requirements.txt",
            "Dockerfile",
            "docker-compose.yml",
            ".github/workflows",
        ]
        structure["config_files"] = {}

        for config_file in config_files:
            config_path = os.path.join(repo_path, config_file)
            structure["config_files"][config_file] = os.path.exists(config_path)

        return structure

    def _analyze_testing(self, repo_path: str) -> Dict:
        """Analyze testing setup."""
        testing = {}

        # Look for test directories and files
        test_patterns = ["test_*.py", "*_test.py", "tests/", "test/"]
        test_files = []

        for pattern in test_patterns:
            import glob

            full_pattern = os.path.join(repo_path, pattern)
            test_files.extend(glob.glob(full_pattern))

        testing["test_count"] = len(test_files)
        testing["has_tests"] = len(test_files) > 0

        # Check for CI configuration
        workflows_dir = os.path.join(repo_path, ".github", "workflows")
        testing["has_ci_workflows"] = os.path.exists(workflows_dir)

        return testing

    def _analyze_cicd(self, repo_path: str) -> Dict:
        """Analyze CI/CD setup."""
        cicd = {}

        # GitHub Actions
        workflows_dir = os.path.join(repo_path, ".github", "workflows")
        if os.path.exists(workflows_dir):
            workflow_files = os.listdir(workflows_dir)
            cicd["github_actions"] = len(
                [f for f in workflow_files if f.endswith(".yml") or f.endswith(".yaml")]
            )
        else:
            cicd["github_actions"] = 0

        # Docker
        cicd["has_dockerfile"] = os.path.exists(os.path.join(repo_path, "Dockerfile"))
        cicd["has_docker_compose"] = os.path.exists(
            os.path.join(repo_path, "docker-compose.yml")
        )

        return cicd

    def _analyze_documentation(self, repo_path: str) -> Dict:
        """Analyze documentation setup."""
        docs = {}

        doc_files = ["README.md", "docs/", "api.md", "CONTRIBUTING.md", "CHANGELOG.md"]
        docs["documentation_files"] = {}

        for doc_file in doc_files:
            doc_path = os.path.join(repo_path, doc_file)
            docs["documentation_files"][doc_file] = os.path.exists(doc_path)

        return docs

    def _analyze_performance(self, repo_path: str) -> Dict:
        """Analyze performance optimization patterns."""
        perf = {}

        # Check for optimization files
        opt_files = ["requirements.txt", "Dockerfile", "performance.py", "optimize.py"]
        perf["optimization_files"] = {}

        for opt_file in opt_files:
            opt_path = os.path.join(repo_path, opt_file)
            perf["optimization_files"][opt_file] = os.path.exists(opt_path)

        return perf

    def _analyze_code_quality(self, repo_path: str) -> Dict:
        """Analyze code quality indicators."""
        quality = {}

        # Check for linting configs
        lint_files = [".eslintrc.js", ".flake8", "pyproject.toml", "setup.cfg"]
        quality["linting_configs"] = {}

        for lint_file in lint_files:
            lint_path = os.path.join(repo_path, lint_file)
            quality["linting_configs"][lint_file] = os.path.exists(lint_path)

        return quality

    def generate_summary(self, analyses: List[Dict]) -> Dict:
        """Generate summary statistics from multiple repository analyses."""
        summary = {
            "total_repositories": len(analyses),
            "average_test_files": sum(a["testing"]["test_count"] for a in analyses)
            / len(analyses)
            if analyses
            else 0,
            "repositories_with_ci": sum(
                1 for a in analyses if a["testing"]["has_ci_workflows"]
            ),
            "repositories_with_docker": sum(
                1 for a in analyses if a["cicd"]["has_dockerfile"]
            ),
            "common_patterns": self._extract_common_patterns(analyses),
        }
        return summary

    def _extract_common_patterns(self, analyses: List[Dict]) -> List[str]:
        """Extract common patterns across analyzed repositories."""
        patterns = []

        # Common testing frameworks
        test_frameworks = set()
        for analysis in analyses:
            # This is a simplified version - in production, would parse actual test files
            if analysis["testing"]["has_ci_workflows"]:
                test_frameworks.add("GitHub Actions")

        patterns.extend([f"Testing: {fw}" for fw in test_frameworks])

        return patterns

    def clone_and_analyze(self, repo_url: str, clone_dir: str) -> Dict:
        """Clone repository and analyze it."""
        import subprocess
        import tempfile

        try:
            # Clone to temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                clone_path = os.path.join(temp_dir, "repo")
                subprocess.run(
                    ["git", "clone", repo_url, clone_path],
                    check=True,
                    capture_output=True,
                )

                analysis = self.analyze_repository(clone_path)
                return analysis

        except Exception as e:
            print(f"Error cloning {repo_url}: {e}")
            return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SOTA repositories for best practices"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="brain tumor classification deep learning",
        help="Search query for repositories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis_report.json",
        help="Output file for analysis results",
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print summary statistics"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum number of repositories to analyze",
    )

    args = parser.parse_args()

    # Get GitHub token from environment
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("Warning: GITHUB_TOKEN not set. API rate limits will apply.")

    analyzer = RepositoryAnalyzer(github_token)

    print(f"Searching for repositories: {args.query}")
    repositories = analyzer.search_repositories(args.query, args.max_results)

    if not repositories:
        print("No repositories found")
        return

    print(f"Found {len(repositories)} repositories")

    analyses = []
    for i, repo in enumerate(repositories, 1):
        print(f"\n[{i}/{len(repositories)}] Analyzing: {repo['name']}")
        analysis = analyzer.clone_and_analyze(repo["clone_url"], f"repo_{i}")
        if "error" not in analysis:
            analyses.append(analysis)
        time.sleep(1)  # Rate limiting

    # Generate report
    report = {
        "metadata": {
            "query": args.query,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_repos_analyzed": len(analyses),
        },
        "repositories": repositories,
        "analyses": analyses,
        "summary": analyzer.generate_summary(analyses),
    }

    # Save report
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nAnalysis complete! Report saved to: {args.output}")

    if args.summary:
        print("\n=== SUMMARY ===")
        summary = report["summary"]
        print(f"Total repositories analyzed: {summary['total_repositories']}")
        print(f"Average test files per repo: {summary['average_test_files']:.1f}")
        print(f"Repositories with CI/CD: {summary['repositories_with_ci']}")
        print(f"Repositories with Docker: {summary['repositories_with_docker']}")
        print("Common patterns:")
        for pattern in summary["common_patterns"]:
            print(f"  - {pattern}")


if __name__ == "__main__":
    main()
