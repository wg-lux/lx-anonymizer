{ pkgs, lib, config, inputs, buildInputs, ... }:
let
  appName = "lx_anonymizer";
  buildInputs = with pkgs; [
    python312Full
    stdenv.cc.cc
    git
    direnv
    glib
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
  dotenv.enable = false;
  dotenv.disableHint = true;

  packages = with pkgs; [
    git
    cudaPackages.cuda_nvcc
    stdenv.cc.cc
    glib
    tesseract
    ollama
    python3Packages.pip
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
  };

  languages.python = {
    enable = true;
    uv = {
      enable = true;
      sync.enable = true;
    };
  };

  scripts.hello.exec = "${pkgs.uv}/bin/uv run python hello.py";

  scripts.env-setup.exec = ''
    export LD_LIBRARY_PATH="${
      with pkgs;
      lib.makeLibraryPath buildInputs
    }:/run/opengl-driver/lib:/run/opengl-driver-32/lib"
    export OLLAMA_BIN=/nix/store/nrcs8aijwjwq450chf1qlm9xxcp8n0iw-ollama-0.6.5/bin/ollama

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
    . .devenv/state/venv/bin/activate
    uv sync

    hello
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.ffmpeg_6}/lib

    cd lx_anonymizer
  '';
}
