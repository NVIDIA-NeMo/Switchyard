.PHONY: env html publish live clean help

REPO_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))..

help:
	@echo "make env      : sync the docs dependency group via uv"
	@echo "make html     : build HTML site (incremental)"
	@echo "make publish  : build HTML site (strict, fails on warnings)"
	@echo "make live     : live-reload server at http://127.0.0.1:8000"
	@echo "make clean    : remove build artefacts"

env:
	cd $(REPO_ROOT) && uv sync --only-group docs

html: env
	cd $(REPO_ROOT) && uv run --only-group docs mkdocs build

publish: env
	cd $(REPO_ROOT) && uv run --only-group docs mkdocs build --strict

live: env
	cd $(REPO_ROOT) && uv run --only-group docs mkdocs serve

clean:
	rm -rf $(REPO_ROOT)/site
