PYTHON ?= python
TAG ?=

.PHONY: test coverage ci publish-gpu-image test-docker-image proto

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

coverage:
	$(PYTHON) -m coverage erase
	$(PYTHON) -m coverage run -m unittest discover -s tests -p 'test_*.py'
	$(PYTHON) -m coverage report -m

ci: coverage

test-docker-image:
	./scripts/test-docker-image.sh

proto:
	$(PYTHON) scripts/generate_grpc_stubs.py

publish-gpu-image:
	./scripts/docker-gh-deploy.sh $(TAG)
