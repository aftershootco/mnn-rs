{
  description = "A simple rust flake using rust-overlay and craneLib";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crane = {
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    mnn-overlay = {
      url = "github:uttarayan21/mnn-nix-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    crane,
    flake-utils,
    nixpkgs,
    rust-overlay,
    mnn-overlay,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            rust-overlay.overlays.default
            (final: prev: {
              mnn = mnn-overlay.packages.${system}.mnn.override {buildConverter = true;};
            })
          ];
        };
        inherit (pkgs) lib;

        stableToolchain = pkgs.rust-bin.stable.latest.default;
        stableToolchainWithRustAnalyzer = pkgs.rust-bin.stable.latest.default.override {
          extensions = ["rust-src" "rust-analyzer"];
          # targets = [
          #   "aarch64-apple-darwin"
          #   "aarch64-apple-ios"
          #   "wasm32-unknown-unknown"
          # ];
        };
        rustToolchain = pkgs.rust-bin.nightly.latest.default.override {extensions = ["rust-src"];};
        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
        src = ./.;
        MNN_SRC = pkgs.fetchFromGitHub {
          owner = "alibaba";
          repo = "MNN";
          rev = "e6042e5e00ba4f6398a5cd5a3615b9f62501438e";
          hash = "sha256-esHU+ociPi7qxficXU0dL+R5MXsblMocrNRgp79hWkk=";
        };
        commonArgs = {
          inherit src MNN_SRC;
          pname = "inspect";
          # cargoExtraArgs = "--example inspect";
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          # BINDGEN_EXTRA_CLANG_ARGS = "-I${pkgs.llvmPackages.libclang.lib}/lib/clang/18/include";
          nativeBuildInputs = with pkgs; [
            cmake
            llvmPackages.libclang.lib
          ];
          buildInputs = with pkgs; [
            darwin.apple_sdk.frameworks.OpenCL
            darwin.apple_sdk.frameworks.OpenGL
            darwin.apple_sdk.frameworks.CoreML
            darwin.apple_sdk.frameworks.Metal
          ];
        };
        cargoArtifacts = craneLib.buildPackage commonArgs;
      in {
        packages = rec {
          inspect = craneLib.buildPackage (commonArgs
            // {
              inherit cargoArtifacts;
              cargoExtraArgs = "--example inspect";
            });
          default = inspect;
        };

        devShells = {
          default = pkgs.mkShell {
            packages = with pkgs; [
              mnn
              darwin.apple_sdk.frameworks.OpenCL
              darwin.apple_sdk.frameworks.OpenGL
              darwin.apple_sdk.frameworks.CoreML
              darwin.apple_sdk.frameworks.Metal
              stableToolchainWithRustAnalyzer
              cargo-nextest
            ];
          };
        };
      }
    );
}
