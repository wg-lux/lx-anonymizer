{ pkgs, lib, config, inputs, ... }:
let
  appName = "lx_anonymizer";
  buildInputs = with pkgs; [
    python312Full
    stdenv.cc.cc
    git
    direnv
    glib
    ollama
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
  ];

  env = {
    LD_LIBRARY_PATH = "${
      with pkgs;
      lib.makeLibraryPath buildInputs
    }:/run/opengl-driver/lib:/run/opengl-driver-32/lib";
    OLLAMA_HOST = "0.0.0.0";
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
  '';

  processes = {
    ollama-serve.exec = "export OLLAMA_DEBUG=1 && ollama serve";
    ollama-pull-model.exec = "ollama pull deepseek-r1:1.5b&";
    ollama-run-model.exec = "ollama run deepseek-r1:1.5b";
    ollama-verify.exec = "curl http://127.0.0.1:11434/api/models";
    };

  tasks = {
  };


  enterShell = ''
    . .devenv/state/venv/bin/activate
    uv sync

    hello
    cd lx_anonymizer
  '';
}
