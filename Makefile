.PHONY: commit quality style

check_dirs := prm800k scripts setup.py

commit:
	pre-commit install
	pre-commit run --all-files

quality:
	ruff check $(check_dirs)
	ruff format --check $(check_dirs)

style:
	ruff check $(check_dirs) --fix
	ruff format $(check_dirs)
