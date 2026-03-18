{
  description = "Pure Nix packaging for lx-anonymizer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nixtest.url = "gitlab:TECHNOFAB/nixtest?dir=lib";
  };

  outputs =
    inputs@{
      self,
      nixpkgs,
      flake-utils,
      nixtest,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        ntlib = nixtest.lib { inherit pkgs; };

        base = pkgs.callPackage ./package.nix { };
        llm = pkgs.callPackage ./package.nix { withLlm = true; };
        ocr = pkgs.callPackage ./package.nix { withOcr = true; };
        full = pkgs.callPackage ./package.nix {
          withLlm = true;
          withOcr = true;
          withNlu = true;
        };
        nativeDrv = pkgs.rustPlatform.buildRustPackage {
          pname = "rust_lx-anonymizer";
          version = "0.1.0";
          src = ./.;
          cargoLock.lockFile = ./Cargo.lock;
          cargoBuildFlags = [ "--lib" ];
          doCheck = false;
        };
        nativeLibDrv = pkgs.lib.getLib nativeDrv;
        nativePythonExt = pkgs.runCommand "lx-anonymizer-native-0.1.0" { } ''
          mkdir -p "$out/lib/python3.12/site-packages/lx_anonymizer"
          native_lib="$(find -L ${nativeLibDrv}/lib -type f -name 'lib_lx_anonymizer_native*.so' | head -n 1)"
          test -n "$native_lib"
          cp "$native_lib" "$out/lib/python3.12/site-packages/lx_anonymizer/_lx_anonymizer_native.so"
        '';
        withNative = pkgs.symlinkJoin {
          name = "lx-anonymizer-with-native";
          paths = [
            base
            nativePythonExt
          ];
        };

        nixtestSuite = ntlib.mkNixtest {
          modules = ntlib.autodiscover { dir = ./nix/tests; };
          args = {
            inherit pkgs ntlib;
            lxAnonymizer = base;
          };
        };
      in
      {
        packages = {
          default = base;
          lx-anonymizer = base;
          lx-anonymizer-with-native = withNative;
          lx-anonymizer-llm = llm;
          lx-anonymizer-ocr = ocr;
          lx-anonymizer-full = full;
          nixtest = nixtestSuite;
        };

        apps.default = {
          type = "app";
          program = "${base}/bin/lx-anonymizer";
        };

        checks = {
          lx-anonymizer = base;
          nixtest = nixtestSuite;
        };
      }
    );
}
