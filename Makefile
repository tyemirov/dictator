PYTHON ?= python3
TAG ?=

.PHONY: test coverage ci publish-gpu-image test-docker-image proto proto-python proto-go proto-check

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

coverage:
	$(PYTHON) -m coverage erase
	$(PYTHON) -m coverage run -m unittest discover -s tests -p 'test_*.py'
	$(PYTHON) -m coverage report -m

ci: proto-check coverage

test-docker-image:
	./scripts/test-docker-image.sh

proto-python:
	$(PYTHON) scripts/generate_grpc_stubs.py

proto-go:
	./scripts/generate_go_grpc_stubs.sh

proto: proto-python proto-go

proto-check:
	@$(MAKE) proto
	@generated_status="$$(git status --porcelain -- dictator/speech/v1 sdk/go/dictatorspeechv1)"; \
	if [ -n "$$generated_status" ]; then \
		echo "Error: generated gRPC stubs are out of sync. Run 'make proto' and commit the changes." >&2; \
		echo "$$generated_status"; \
		exit 1; \
	fi

publish-gpu-image:
	./scripts/docker-gh-deploy.sh $(TAG)
