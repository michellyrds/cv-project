
.PHONY: environment
environment:
	pyenv install -s 3.10.5
	pyenv virtualenv 3.10.5 cv-project
	pyenv local cv-project

.PHONY: requirements
requirements:
	pip install -Ur requirements.txt
