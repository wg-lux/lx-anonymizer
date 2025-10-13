# Releasing LX Anonymizer

This guide describes how to cut a new release to PyPI and GitHub.

## Prerequisites
- PyPI account with maintainer access and an API token stored as `PYPI_API_TOKEN` in repo secrets.
- GitHub personal access to create tags/releases.
- Ability to run CI workflows (GitHub Actions).
- Local environment with `uv`, `hatchling`, and `twine` available (`uv pip install build twine`).

## Release checklist

1. **Sync main**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Update version**
   - Edit `pyproject.toml` with the new semantic version (e.g., `1.0.0`).
   - Update `CHANGELOG.md`:
     - Move items from `[Unreleased]` into a new section `[vX.Y.Z] - YYYY-MM-DD`.
     - Add an empty `[Unreleased]` header.
   - Verify `README.md` references the correct version if applicable.

3. **Run preflight checks**
   ```bash
   uv sync
   uv run flake8
   uv run pytest -m "not gpu"
   uv run python -m build
   ```
   Optionally run GPU/LLM markers if the change affects those paths.

4. **Test the wheel locally**
   ```bash
   uv pip install dist/lx_anonymizer-<version>-py3-none-any.whl
   uv run python -m cli.report_reader --help
   ```

5. **Publish to TestPyPI (optional but recommended)**
   ```bash
   uv run python -m twine upload --repository testpypi dist/*
   ```
   Install from TestPyPI in a clean virtual environment to verify:
   ```bash
   python -m venv /tmp/lxa-test
   source /tmp/lxa-test/bin/activate
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple lx-anonymizer
   lx-anonymizer --help
   ```

6. **Commit and tag**
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "chore(release): vX.Y.Z"
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin main --tags
   ```

7. **Create GitHub release**
   - Open https://github.com/wg-lux/lx-anonymizer/releases/new
   - Select the pushed tag `vX.Y.Z`
   - Title: `vX.Y.Z`
   - Release notes: paste the changelog entry
   - Attach `dist/*.whl` and `dist/*.tar.gz`
   - Publish release

8. **Publish to PyPI**
   - Ensure `PYPI_API_TOKEN` is configured in GitHub Actions secrets.
   - Trigger the `publish.yml` workflow by pushing the tag (step 6). The workflow will build and upload the distribution. Alternatively, publish manually:
   ```bash
   uv run python -m twine upload dist/*
   ```

9. **Post-release**
   - Bump `pyproject.toml` to the next patch prerelease (e.g., `1.0.1-dev`) and update `[Unreleased]` section with placeholders.
   - Close issues fixed in the release, label accordingly.
   - Announce release (Slack, mailing list, etc.).

## Troubleshooting
- **Build failures**: Delete `dist/` and `build/`, reinstall dependencies, re-run `python -m build`.
- **Missing assets**: Ensure `MANIFEST.in` covers all required data files (`names_dict/`, `devices/`).
- **PyPI upload permission denied**: Regenerate the API token and update the GitHub secret.

Feel free to automate additional steps as the release process matures.
