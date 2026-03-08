PYTHON ?= python
TAG ?=

.PHONY: test coverage ci publish-gpu-image

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

coverage:
	$(PYTHON) -m coverage erase
	$(PYTHON) -m coverage run -m unittest discover -s tests -p 'test_*.py'
	$(PYTHON) -m coverage report -m

ci: coverage

publish-gpu-image:
	./scripts/docker-gh-deploy.sh $(TAG)
