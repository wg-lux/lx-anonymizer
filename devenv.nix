{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:
let
  python = pkgs.python312;
  rustSrc = lib.cleanSourceWith {
    src = ./.;
    filter =
      path: type:
      let
        rel = lib.removePrefix "${toString ./.}/" (toString path);
        base = builtins.baseNameOf (toString path);
        ignoredBaseNames = [
          ".devenv"
          ".direnv"
          ".env"
          ".git"
          ".mypy_cache"
          ".pytest_cache"
          ".ruff_cache"
          ".venv"
          "__pycache__"
          "result"
          "target"
        ];
        ignoredPrefixes = [
          "dist/"
          "logs/"
        ];
      in
      !(builtins.elem base ignoredBaseNames || lib.any (prefix: lib.hasPrefix prefix rel) ignoredPrefixes);
  };

  buildInputs = with pkgs; [
    python
    python312Packages.tkinter
    stdenv.cc.cc
    git
    direnv
    glib
    zlib
    libglvnd
    ollama
    cmake
    gcc
    pkg-config
    protobuf
    libGL
  ];

  runtimePackages = with pkgs; [
    git
    cudaPackages.cuda_nvcc
    stdenv.cc.cc
    glib
    zlib
    (tesseract.override {
      enableLanguages = [
        "eng"
        "deu"
      ];
    })
    ollama
    uv
    python312Packages.pip
    libglvnd
    cmake
    gcc
    pkg-config
    protobuf
    python312Packages.sentencepiece
    ffmpeg_6-headless
    maturin
    cargo
    rustc
  ];
in
{
  dotenv.enable = true;
  dotenv.disableHint = false;

  languages.python = {
    enable = true;
    version = "3.12";
    uv = {
      enable = true;
      sync.enable = true;
    };
  };

  languages.rust.enable = true;

  packages = lib.unique (buildInputs ++ runtimePackages);

  # devenv 2 outputs only become useful for Python once pyproject-nix is added
  # via `devenv inputs add pyproject-nix github:pyproject-nix/pyproject.nix --follows nixpkgs`.
  outputs =
    lib.optionalAttrs (inputs ? pyproject-nix) (
      let
        pythonApp = config.languages.python.import ./. { };
        nativeDrv = config.languages.rust.import ./. { };
        nativeLibDrv = lib.getLib nativeDrv;

        nativeApp = pkgs.runCommand "lx-anonymizer-native-0.1.0" { } ''
          mkdir -p "$out/${python.sitePackages}/lx_anonymizer"
          native_lib="$(find -L ${nativeLibDrv}/lib -type f -name 'lib_lx_anonymizer_native*.so' | head -n 1)"
          test -n "$native_lib"
          cp "$native_lib" "$out/${python.sitePackages}/lx_anonymizer/_lx_anonymizer_native.so"
        '';
      in
      {
        python = pythonApp;
        app = pythonApp;
      }
    );

  env = {
    LD_LIBRARY_PATH =
      with pkgs;
      lib.makeLibraryPath runtimePackages
      + ":/run/opengl-driver/lib:/run/opengl-driver-32/lib"
      + ":/usr/lib/wsl/lib"
      + ":/usr/lib/x86_64-linux-gnu"
      + ":/usr/lib";
    OLLAMA_HOST = "0.0.0.0";
    PYTORCH_ALLOC_CONF = "expandable_segments:True";
    PYO3_PYTHON = "${python}/bin/python";
    UV_PYTHON = lib.mkForce "${python}/bin/python";
  };

  scripts.hello.exec = "${pkgs.uv}/bin/uv run python hello.py";

  scripts.env-setup.exec = ''
    export LD_LIBRARY_PATH="${
      with pkgs; lib.makeLibraryPath buildInputs
    }:/run/opengl-driver/lib:/run/opengl-driver-32/lib"
    export TESSDATA_PREFIX="${
      pkgs.tesseract.override {
        enableLanguages = [
          "eng"
          "deu"
        ];
      }
    }/share"
  '';

  scripts.uvs.exec = ''
    uv sync --extra dev --extra ocr --extra llm
  '';

  processes = {
    ollama-pull-deepseek-model.exec = "ollama pull deepseek-r1:1.5b&";
    ollama-run-deepseek-model.exec = "ollama run deepseek-r1:1.5b";
    ollama-verify.exec = "curl http://127.0.0.1:11434/api/models";
  };

  tasks = {
    "ollama:serve".exec = "export OLLAMA_DEBUG=1 && ollama serve";
  };

  enterShell = ''

    if [ ! -d ".devenv/state/venv" ]; then
       export SYNC_CMD='uv sync --extra dev --extra ocr --extra llm'

       echo "Virtual environment not found. Running initial uv sync..."
       $SYNC_CMD || echo "Error: Initial uv sync failed. Please check network and pyproject.toml."
    else
       echo "Syncing Python dependencies with uv..."
       $SYNC_CMD --quiet || echo "Warning: uv sync failed. Environment might be outdated."
    fi

    ACTIVATED=false
    if [ -f ".devenv/state/venv/bin/activate" ]; then
      source .devenv/state/venv/bin/activate
      ACTIVATED=true
      echo "Virtual environment activated."
    else
      echo "Warning: uv virtual environment activation script not found. Run 'devenv task run env:clean' and re-enter shell."
    fi

    echo "Exporting environment variables from .env file..."
    if [ -f ".env" ]; then
      set -a
      source .env
      set +a
      echo ".env file loaded successfully."
    elif [ -f "local_settings.py" ]; then
      echo "Detected luxnix managed environment - using system environment variables"
      echo "No .env file needed"
    else
      echo "Warning: .env file not found. Please run 'devenv tasks run env:build' to create it."
    fi
    env-setup
  '';
}
