.PHONY: docs docs-serve docs-clean test

# Suppress third-party DeprecationWarnings that fire at plugin-import time
# (mkdocs_gallery / jupyter_core) and cannot be filtered via mkdocs.yml.
MKDOCS_ENV = JUPYTER_PLATFORM_DIRS=1 \
             PYTHONWARNINGS="ignore::DeprecationWarning:mkdocs_gallery,\
ignore::DeprecationWarning:jupyter_core,\
ignore::DeprecationWarning:jupyter_client,\
ignore::DeprecationWarning:mkdocs"

docs:
	$(MKDOCS_ENV) mkdocs build

docs-serve:
	$(MKDOCS_ENV) mkdocs serve

docs-clean:
	rm -rf docs/generated site

test:
	pytest
