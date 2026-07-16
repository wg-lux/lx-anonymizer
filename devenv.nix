{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:
let
  python = pkgs.python312;

  # Avoid repeated inline evaluations by binding the override once
  tesseractCustom = pkgs.tesseract.override {
    enableLanguages = [
      "eng"
      "deu"
    ];
  };

  # 1. Pure C-libraries that need to be in LD_LIBRARY_PATH
  libs = with pkgs; [
    stdenv.cc.cc
    glib
    zlib
    libglvnd
    libGL
    libxcb
  ];

  # 2. Build-time tools (DO NOT evaluate library paths for these)
  buildTools = with pkgs; [
    git
    direnv
    cmake
    gcc
    pkg-config
    protobuf
    cargo
    rustc
    maturin
    uv
  ];

  # 3. Rest of the runtime/shell packages
  otherPackages = with pkgs; [
    python
    python312Packages.pip
    python312Packages.tkinter
    python312Packages.sentencepiece
    ollama
    cudaPackages.cuda_nvcc
    tesseractCustom
    ffmpeg_6-headless
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
      sync = {
        enable = true;
        extras = [
          "dev"
          "evaluation"
        ];
      };
    };
  };

  languages.rust.enable = true;

  # Avoid lib.unique. Concatenating lists directly is instant.
  packages = buildTools ++ libs ++ otherPackages;

  # pyproject-nix & rust imports are only evaluated if outputs are specifically requested
  outputs = lib.optionalAttrs (inputs ? pyproject-nix) (
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
    # Point only to the lightweight 'libs' list.
    # This prevents Nix from walking the recursive trees of gcc, rustc, and cuda_nvcc.
    LD_LIBRARY_PATH =
      "/run/opengl-driver/lib:/run/opengl-driver-32/lib"
      + ":/usr/lib/wsl/lib"
      + ":/usr/lib/x86_64-linux-gnu"
      + ":/usr/lib"
      + ":${lib.makeLibraryPath libs}";

    OLLAMA_HOST = "127.0.0.1:11434";
    PYTORCH_ALLOC_CONF = "expandable_segments:True";
    PYO3_PYTHON = "${python}/bin/python";
    UV_PYTHON = lib.mkForce "${python}/bin/python";
  };

  scripts.hello.exec = "${pkgs.uv}/bin/uv run python hello.py";

  scripts.env-setup.exec = ''
    export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver-32/lib:${lib.makeLibraryPath libs}"
    export TESSDATA_PREFIX="${tesseractCustom}/share"
  '';

  scripts.uvs.exec = ''
    uv sync --extra dev --extra gpu
  '';

  processes = {
    ollama-gemma4-provision.exec = "bash scripts/provision_ollama_gemma4.sh";
    ollama-verify.exec = "curl --fail http://127.0.0.1:11434/api/tags";
  };

  tasks = {
    "ollama:serve".exec = "export OLLAMA_DEBUG=1 && ollama serve";
    "ollama:provision-gemma4".exec = "bash scripts/provision_ollama_gemma4.sh";
  };

  enterShell = ''

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
