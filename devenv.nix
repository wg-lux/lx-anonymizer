{ pkgs, lib, config, inputs, buildInputs, ... }:
let
  appName = "lx_anonymizer";
  buildInputs = with pkgs; [
    python311Full
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
    UV_PYTHON_DOWNLOADS = "managed";
    UV_PYTHON_PREFERENCE = "system";
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
    uv sync
    uv run python env_setup.py
    hello
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.ffmpeg_6}/lib

    cd lx_anonymizer
  '';
}
