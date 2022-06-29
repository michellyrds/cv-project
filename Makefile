.DEFAULT_GOAL := help


.PHONY: environment
environment: ## setup project environment
	pyenv install -s 3.7.13
	pyenv virtualenv 3.7.13 cv-project
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

.PHONY: start-dev
start-dev: ## run application in dev mode
	streamlit run main.py --logger.level=debug
	
.PHONY: docker-build
docker-build: ## setup project using docker container
	docker build -t streamlitapp:latest .
	docker run -p 8501:8501 streamlitapp:latest

.PHONY: docker-start
docker-start: ## start application server on docker
	docker run -p 8501:8501 streamlitapp:latest


help: ## Prompts help for every command
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
    awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

