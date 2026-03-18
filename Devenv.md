# What `devenv.nix` Does

The repository uses [devenv](https://devenv.sh/) for two related jobs:

- creating a development shell with Python, Rust, OCR, ffmpeg, and Ollama tooling
- exposing build outputs that other flakes can consume

The important part is the `outputs` block in [devenv.nix](/home/admin/lx-anonymizer/devenv.nix).
When the caller provides `pyproject-nix`, devenv can export package-like outputs
instead of only a shell.

In this repository, `devenv.nix` builds:

- `python`: the Python application imported from the local `pyproject.toml`
- `native`: a package containing the compiled Rust extension copied into
  `site-packages/lx_anonymizer/_lx_anonymizer_native.so`
- `native-raw`: the raw Rust build output
- `app`: a `symlinkJoin` of the Python package and native extension

That `app` output is what makes the project convenient to consume as a flake
input from another repo.

## Using This Repo As A Flake Input

Add the repository to another project's `devenv.yaml`:

```yaml
inputs:
  lx-anonymizer:
    url: github:wg-lux/lx-anonymizer
  pyproject-nix:
    url: github:pyproject-nix/pyproject.nix
    inputs:
      nixpkgs:
        follows: nixpkgs
```

Then use it from that project's `devenv.nix`:

```nix
{ inputs, ... }:
{
  packages = [
    inputs.lx-anonymizer.outputs.${builtins.currentSystem}.packages.default
  ];
}
```

If you want the package produced by this repo's `devenv.nix` outputs rather than
the plain `flake.nix` package set, import the devenv output explicitly:

```nix
{ inputs, pkgs, ... }:
let
  lx = inputs.lx-anonymizer.outputs.${pkgs.system};
in
{
  packages = [
    lx.packages.default
  ];
}
```

If your integration is based on devenv outputs, the most relevant values are:

- `config.devenv.outputs.python`
- `config.devenv.outputs.native`
- `config.devenv.outputs.app`

The combined native-enabled package is the `app` output.

## Relationship Between `devenv.nix` And `flake.nix`

This repo currently contains both:

- [devenv.nix](/home/admin/lx-anonymizer/devenv.nix): defines the development shell and
  optional devenv outputs
- [flake.nix](/home/admin/lx-anonymizer/flake.nix): defines standalone flake packages like
  `lx-anonymizer`, `lx-anonymizer-with-native`, `lx-anonymizer-ocr`, and `lx-anonymizer-full`

Use them like this:

- Use `devenv.nix` when another devenv-based project wants this repo as an input.
- Use `flake.nix` when you want standard `nix build .#...` style package outputs.

## What Gets Bundled

### Python layer

The Python application comes from `pyproject.toml` and is the baseline portable
package. Optional features remain optional:

- `[ocr]` for OCR-heavy paths
- `[llm]` for Ollama-backed metadata extraction
- `[nlu]` for Flair-based NER

### Native layer

The Rust extension is optional at runtime. Python imports it through
`lx_anonymizer._native` and falls back to pure Python when the shared object is
missing or only partially implemented.

The native-enabled Nix package works by copying the compiled shared object into:

```text
.../site-packages/lx_anonymizer/_lx_anonymizer_native.so
```

That is why the combined devenv output and the `lx-anonymizer-with-native`
package behave differently from a plain PyPI install.
