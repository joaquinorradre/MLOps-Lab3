install:
	pip install uv &&\
	uv sync

test:
	uv run python -m pytest -vv --cov=mylib --cov=api --cov=cli tests/

format:
	uv run black api/*.py cli/*.py  mylib/*.py  tests/*.py

lint:
	uv run python -m pylint --rcfile=.pylintrc --ignore-patterns=test_.*\.py api/*.py cli/*.py mylib/*.py tests/*.py

all: install format lint test