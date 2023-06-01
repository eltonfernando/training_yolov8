
requi:
	pip3 freeze > requirements.txt

auto_formater:
	pre-commit run --all-files -c .pre-commit-config.yaml
