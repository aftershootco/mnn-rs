{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
        };
      in {
        packages.venv = pkgs.stdenv.mkDerivation {
          src = ./.;
          name = "venv";
          buildInputs = [(pkgs.python311.withPackages (ps: [ps.pip ps.virtualenv]))];
          buildPhase = ''
            virtualenv -p python3 venv
            source venv/bin/activate
            pip install tensorflow tf2onnx onnxruntime
          '';
          installPhase = ''
            mkdir -p $out
            cp -r venv $out
          '';
        };
        devShell = pkgs.mkShell {
          packages = [(pkgs.python311.withPackages (ps: [ps.pip ps.onnx ps.onnxruntime]))];
        };
      }
    );
}
