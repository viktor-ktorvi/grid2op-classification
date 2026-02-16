pre-commit:
	uv run pre-commit run --all-files

mypy:
	uv run mypy .

check-everything: pre-commit mypy