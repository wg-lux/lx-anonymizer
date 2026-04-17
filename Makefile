YTHON_BIN ?= python
MATURIN_BIN ?= maturin
TWINE_BIN ?= twine
PYPI_DIST_DIR ?= dist
PYPI_MANIFEST_PATH ?= Cargo.toml
PYPI_INTERPRETER ?=
PYPI_COMPATIBILITY ?= linux

pypi-wheel:
	@echo "Building Rust Backend Extension and Main Django Wheel with Zig/Manylinux..."
	@mkdir -p $(PYPI_DIST_DIR)
	# Unset the Nix-specific platform override to let Maturin/Zig do their job
	unset _PYTHON_HOST_PLATFORM; \
	$(MATURIN_BIN) build --release \
		--zig \
		--compatibility manylinux2014 \
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