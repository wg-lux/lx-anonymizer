{ pkgs, lib, config, inputs, ... }:
let
  buildInputs = with pkgs; [
    python312Full
    # cudaPackages.cuda_cudart
    # cudaPackages.cudnn
    stdenv.cc.cc
    glib
  ];


in 
{

  # A dotenv file was found, while dotenv integration is currently not enabled.
  dotenv.enable = false;
  dotenv.disableHint = true;


  packages = with pkgs; [
    cudaPackages.cuda_nvcc
    stdenv.cc.cc
    glib
    tesseract
    python3Packages.pip
  ];

  env = {
    LD_LIBRARY_PATH = "${
      with pkgs;
      lib.makeLibraryPath buildInputs
    }:/run/opengl-driver/lib:/run/opengl-driver-32/lib";

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
    # django.exec = "run-dev-server";
    silly-example.exec = "while true; do echo hello && sleep 10; done";
    test-main.exec = "python lx_anonymizer/main.py -i lx_anonymizer/test_images/namen.jpg";
    # django.exec = "${pkgs.uv}/bin/uv run python manage.py runserver 127.0.0.1:8123";
  };
  tasks."run-test-main" = {
    exec = "devenv up test-main";
    after = [ "devenv:python:virtualenv" ];
  };

  enterShell = ''
    . .devenv/state/venv/bin/activate
    hello
    cd lx_anonymizer
  '';
}
