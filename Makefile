.DEFAULT_GOAL := help

.PHONY: venv
venv: ## Create the virtual environment
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@uv venv


.PHONY: install
install: venv ## Install a virtual environment
	@uv pip install --upgrade pip
	@uv pip install -r requirements.txt


.PHONY: fmt
fmt: venv ## Run autoformatting and linting
	@uv pip install pre-commit
	@uv run pre-commit install
	@uv run pre-commit run --all-files


.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@git clean -X -d -f
	@git branch -v | grep "\[gone\]" | cut -f 3 -d ' ' | xargs git branch -D


.PHONY: help
help:  ## Display this help screen
	@echo -e "\033[1mAvailable commands:\033[0m"
	@grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort


.PHONY: marimo
marimo: install ## Install Marimo
	@uv pip install marimo
	@uv run marimo edit notebooks