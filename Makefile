ifneq ("$(wildcard .venv/bin/python)","")
PYTHON ?= .venv/bin/python
else ifneq ("$(shell command -v python3.11 2>/dev/null)","")
PYTHON ?= python3.11
else
PYTHON ?= python3
endif

PROTO_PYTHON_VENV ?= tools/proto-python
PROTO_PYTHON := $(PROTO_PYTHON_VENV)/bin/python
PROTO_PYTHON_READY := $(PROTO_PYTHON_VENV)/.ready
TAG ?=

.PHONY: test coverage ci publish-gpu-image test-docker-image proto proto-python proto-go proto-check proto-python-tools

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

coverage:
	$(PYTHON) -m coverage erase
	$(PYTHON) -m coverage run -m unittest discover -s tests -p 'test_*.py'
	$(PYTHON) -m coverage report -m

ci: proto-check coverage

test-docker-image:
	./scripts/test-docker-image.sh

proto-python: proto-python-tools
	$(PROTO_PYTHON) scripts/generate_grpc_stubs.py

proto-python-tools: $(PROTO_PYTHON_READY)

$(PROTO_PYTHON_READY):
	@bootstrap_python="$$(command -v python3.11 || command -v python3)"; \
	if [ -z "$$bootstrap_python" ]; then \
		echo "Error: python3.11 or python3 is required to bootstrap protobuf tools." >&2; \
		exit 1; \
	fi; \
	rm -rf "$(PROTO_PYTHON_VENV)"; \
	"$$bootstrap_python" -m venv "$(PROTO_PYTHON_VENV)"; \
	"$(PROTO_PYTHON)" -m pip install --upgrade pip >/dev/null; \
	"$(PROTO_PYTHON)" -m pip install grpcio-tools protobuf >/dev/null; \
	touch "$(PROTO_PYTHON_READY)"

proto-go:
	./scripts/generate_go_grpc_stubs.sh

proto: proto-python-tools proto-python proto-go

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
