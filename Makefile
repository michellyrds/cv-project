.DEFAULT_GOAL := help


.PHONY: environment
environment: ## setup project environment
	pyenv install -s 3.10.5
	pyenv virtualenv 3.10.5 cv-project
	pyenv local cv-project

.PHONY: requirements
requirements: ## install project dependencies
	pip install -Ur requirements.txt

.PHONY: clean
clean: ## delete python bytecode-compiled files
	find . -type d -name __pycache__ -delete

.PHONY: start
start: ## start application server
	streamlit run main.py

help: ## Prompts help for every command
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
    awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'