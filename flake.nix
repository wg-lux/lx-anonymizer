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
        withNative = base;

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
