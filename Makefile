.PHONY: help install precommit precommit-all format lint typecheck test clean

# Default target
help:
	@echo "Available commands:"
	@echo "  make install        - Install dependencies with poetry"
	@echo "  make precommit      - Run pre-commit hooks on staged files"
	@echo "  make precommit-all  - Run pre-commit hooks on all files"
	@echo "  make format         - Format code with black and isort"
	@echo "  make lint           - Run linting checks"
	@echo "  make typecheck      - Run type checking with mypy"
	@echo "  make test           - Run tests (placeholder)"
	@echo "  make clean          - Clean up temporary files"
	@echo "  make help           - Show this help message"

# Install dependencies
install:
	poetry install

# Run pre-commit hooks on staged files only
precommit:
	poetry run pre-commit run

# Run pre-commit hooks on all files
precommit-all:
	poetry run pre-commit run --all-files

# Format code
format:
	poetry run black density_engine/
	poetry run isort density_engine/

# Run linting (if you add flake8 or other linters later)
lint:
	@echo "Linting not configured yet. Add linters to .pre-commit-config.yaml"

# Run type checking
typecheck:
	poetry run mypy density_engine/

# Run tests (placeholder - add your test command here)
test:
	@echo "Tests not configured yet. Add pytest or your test runner here"

# Clean up temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf build/
	rm -rf dist/

# Development setup
dev-setup: install
	poetry run pre-commit install
	@echo "Development environment ready!"
	@echo "Run 'make precommit' to check your changes before committing"
