{ pkgs, lib, config, inputs, buildInputs, ... }:
let
  appName = "lx_anonymizer";
  buildInputs = with pkgs; [
    # python312
    python312Packages.tkinter
    stdenv.cc.cc
    git
    direnv
    glib
    zlib
    libglvnd
    ollama
    cmake          # build system
    gcc            # C/C++ compiler tool-chain
    pkg-config
    protobuf
  ];

  customTasks = (
    import ./devenv/tasks/default.nix ({
      inherit config pkgs lib;
    })
  );

in
{
  dotenv.enable = true;
  dotenv.disableHint = false;

  languages.python = {
    enable = true;
    package = pkgs.python3.withPackages(ps: with ps; [tkinter]); #known devenv issue with python3Packages since python3Full was deprecated
    uv = {
      enable = true;
      sync.enable = true;
    };
  };
  

  packages = with pkgs; [
    git
    cudaPackages.cuda_nvcc
    stdenv.cc.cc
    glib
    zlib
    (tesseract.override {
      enableLanguages = [ "eng" "deu" ];  # English + German traineddata
    })
    ollama
    uv  # Python package manager
    # python312
    python312Packages.tkinter
    python312Packages.pip
    libglvnd
    cmake
    gcc
    pkg-config
    protobuf
    python312Packages.sentencepiece
    ffmpeg_6-headless
  ];

  env = {
    LD_LIBRARY_PATH = "${
      with pkgs;
      lib.makeLibraryPath buildInputs
    }:/run/opengl-driver/lib:/run/opengl-driver-32/lib";
    OLLAMA_HOST = "0.0.0.0";
    PYTORCH_CUDA_ALLOC_CONF= "expandable_segments:True";
    # UV_PYTHON_DOWNLOADS = "never";  # Use system Python from Nix, don't download
    # UV_PYTHON_PREFERENCE = "system";
    # Note: TESSDATA_PREFIX should point to parent of tessdata/ for CLI tools
    # but tesserocr needs the tessdata/ dir itself (handled in Python code)
    TESSDATA_PREFIX = "${pkgs.tesseract.override { enableLanguages = [ "eng" "deu" ]; }}/share";
  };


  scripts.hello.exec = "${pkgs.uv}/bin/uv run python hello.py";

  scripts.env-setup.exec = ''
    export LD_LIBRARY_PATH="${
      with pkgs;
      lib.makeLibraryPath buildInputs
    }:/run/opengl-driver/lib:/run/opengl-driver-32/lib"
    export TESSDATA_PREFIX="${pkgs.tesseract.override { enableLanguages = [ "eng" "deu" ]; }}/share"
  '';



  processes = {
    #ollama-pull-llama.exec = "ollama pull llama3.3";
    #ollama-run-llama.exec = "ollama run llama3.3";
    ollama-pull-deepseek-model.exec = "ollama pull deepseek-r1:1.5b&";
    ollama-run-deepseek-model.exec = "ollama run deepseek-r1:1.5b";
    #ollama-pull-med-model.exec = "ollama pull rjmalagon/medllama3-v20:fp16";
    #ollama-run-med-model.exec = "ollama run rjmalagon/medllama3-v20:fp16";
    ollama-verify.exec = "curl http://127.0.0.1:11434/api/models";
    };

  tasks = {
    "ollama:serve".exec = "export OLLAMA_DEBUG=1 && ollama serve";
  };

  enterShell = ''
    export SYNC_CMD='uv sync --extra dev --extra ocr --extra llm'
    # uv run python env_setup.py # modifies env
       

    # Ensure dependencies are synced using uv
    # Check if venv exists. If not, run sync verbosely. If it exists, sync quietly.
    if [ ! -d ".devenv/state/venv" ]; then
       echo "Virtual environment not found. Running initial uv sync..."
       $SYNC_CMD || echo "Error: Initial uv sync failed. Please check network and pyproject.toml."
    else
       # Sync quietly if venv exists
       echo "Syncing Python dependencies with uv..."
       $SYNC_CMD --quiet || echo "Warning: uv sync failed. Environment might be outdated."
    fi

    # Activate Python virtual environment managed by uv
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
  '';
}
