.PHONY: check install develop lint pre flake test retest untrack

check:
	python setup.py check

install:
	python setup install

develop:
	python setup develop

lint:
	black .
	isort -y
	flake8 .

pre:
	pre-commit run -a

flake:
	flake8 .

untrack:
	git rm -r --cached .
	git add .
	git commit -m ".gitignore fix"