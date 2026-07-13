.PHONY: check preview clean help

FERN_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))fern
FERN_VERSION := $(shell jq -r .version $(FERN_DIR)/fern.config.json)
FERN := npx -y fern-api@$(FERN_VERSION)

help:
	@echo "make check   : validate Fern configuration and content"
	@echo "make preview : serve a local Fern preview"
	@echo "make clean   : remove local Fern build artifacts"

check:
	cd $(FERN_DIR) && $(FERN) check

preview:
	cd $(FERN_DIR) && $(FERN) docs dev

clean:
	rm -rf $(FERN_DIR)/.fern $(FERN_DIR)/node_modules
