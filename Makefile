# Mekong SWAT + SMAP data assimilation.
# Every target is safe to re-run; nothing here overwrites archived data.

PYTHON ?= python3

.PHONY: help install info reproduce test test-fast tune fortran clean

help:                     ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:                  ## Install the package and its test dependencies
	$(PYTHON) -m pip install -e ".[dev]"

info:                     ## Environment and data inventory check
	mekong-da info

reproduce:                ## Run the worked example and verify it (seconds)
	mekong-da reproduce

test:                     ## Run the whole test suite, Fortran build included
	$(PYTHON) -m pytest tests -v

test-fast:                ## Run everything except the Fortran build
	$(PYTHON) -m pytest tests -v -m "not slow"

tune:                     ## Re-run the two-stage parameter search
	mekong-da tune

fortran:                  ## Compile the SWAT-DA executable
	mekong-da build-fortran

clean:                    ## Remove generated results and build products
	rm -rf outputs/enkf outputs/tables outputs/enkf_tuning fortran/bin
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
