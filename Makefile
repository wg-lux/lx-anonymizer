YTHON_BIN ?= python
MATURIN_BIN ?= maturin
TWINE_BIN ?= twine
PYTHON_BIN ?= python

PYPI_DIST_DIR ?= dist
PYPI_MANIFEST_PATH ?= Cargo.toml
PYPI_INTERPRETER ?=
# Local Nix/Zig builds link against the host glibc. Release wheels are built
# in CI with a manylinux container and can use the stricter manylinux target.
PYPI_COMPATIBILITY ?= manylinux_2_34

pypi-wheel:
	@echo "Building Rust Backend Extension and Main Django Wheel with Zig/Manylinux..."
	@mkdir -p $(PYPI_DIST_DIR)
	# Unset the Nix-specific platform override to let Maturin/Zig do their job
	unset _PYTHON_HOST_PLATFORM; \
	$(MATURIN_BIN) build --release \
		--zig \
		--compatibility $(PYPI_COMPATIBILITY) \
		--out $(PYPI_DIST_DIR)

pypi-sdist:
	@mkdir -p $(PYPI_DIST_DIR)
	@set -e; \
	args="--out $(PYPI_DIST_DIR)"; \
	if [ -n "$(PYPI_MATURIN_ARGS)" ]; then \
		args="$$args $(PYPI_MATURIN_ARGS)"; \
	fi; \
	echo "$(MATURIN_BIN) sdist $$args"; \
	$(MATURIN_BIN) sdist $$args

pypi-clean:
	@rm -rf $(PYPI_DIST_DIR) build *.egg-info
	@echo \"Cleaned dist/build artifacts\"

package:
	@make pypi-clean
	@make pypi-wheel
	@make pypi-sdist

.PHONY: lint-structure test-structure
lint-structure:
	$(PYTHON_BIN) scripts/lint_structure.py

test-structure: lint-structure
	.devenv/state/venv/bin/pyright
	uv sync --extra dev
	uv run pytest tests/test_lint_structure.py -q
