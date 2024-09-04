{
  description = "A simple rust flake using rust-overlay and craneLib";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    mnn-overlay = {
      url = "github:uttarayan21/mnn-nix-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    advisory-db = {
      url = "github:rustsec/advisory-db";
      flake = false;
    };
  };

  outputs = {
    crane,
    flake-utils,
    nixpkgs,
    rust-overlay,
    mnn-overlay,
    advisory-db,
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
        stableToolchainWithLLvmTools = pkgs.rust-bin.stable.latest.default.override {
          extensions = ["rust-src" "llvm-tools"];
        };
        stableToolchainWithRustAnalyzer = pkgs.rust-bin.stable.latest.default.override {
          extensions = ["rust-src" "rust-analyzer"];
        };
        craneLib = (crane.mkLib pkgs).overrideToolchain stableToolchain;
        craneLibLLvmTools = (crane.mkLib pkgs).overrideToolchain stableToolchainWithLLvmTools;
        src = ./.;
        MNN_SRC = pkgs.fetchFromGitHub {
          owner = "alibaba";
          repo = "MNN";
          rev = "e6042e5e00ba4f6398a5cd5a3615b9f62501438e";
          hash = "sha256-esHU+ociPi7qxficXU0dL+R5MXsblMocrNRgp79hWkk=";
        };
        commonArgs = {
          inherit src MNN_SRC;
          pname = "mnn";
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
        checks = {
          mnn-clippy = craneLib.cargoClippy (commonArgs
            // {
              inherit cargoArtifacts;
              cargoClippyExtraArgs = "--all-targets -- --deny warnings";
            });
          mnn-docs = craneLib.cargoDoc (commonArgs // {inherit cargoArtifacts;});
          mnn-fmt = craneLib.cargoFmt {inherit src;};
          mnn-toml-fmt = craneLib.taploFmt {
            src = pkgs.lib.sources.sourceFilesBySuffices src [".toml"];
          };
          # Audit dependencies
          mnn-audit = craneLib.cargoAudit {
            inherit src advisory-db;
          };

          # Audit licenses
          mnn-deny = craneLib.cargoDeny {
            inherit src;
          };
          # Run tests with cargo-nextest
          # Consider setting `doCheck = false` on other crate derivations
          # if you do not want the tests to run twice
          mnn-nextest = craneLib.cargoNextest (commonArgs
            // {
              inherit cargoArtifacts;
              partitions = 1;
              partitionType = "count";
            });

          # Ensure that cargo-hakari is up to date
          # mnn-hakari = craneLib.mkCargoDerivation {
          #   inherit src;
          #   pname = "mnn-hakari";
          #   cargoArtifacts = null;
          #   doInstallCargoArtifacts = false;
          #
          #   buildPhaseCargoCommand = ''
          #     cargo hakari generate --diff  # workspace-hack Cargo.toml is up-to-date
          #     cargo hakari manage-deps --dry-run  # all workspace crates depend on workspace-hack
          #     cargo hakari verify
          #   '';
          #
          #   nativeBuildInputs = [
          #     pkgs.cargo-hakari
          #   ];
          # };
        };
        packages =
          rec {
            mnn = craneLib.buildPackage (commonArgs // {inherit cargoArtifacts;});
            inspect = craneLib.buildPackage (commonArgs
              // {
                inherit cargoArtifacts;
                pname = "inspect";
                cargoExtraArgs = "--example inspect";
              });
            default = mnn;
          }
          // lib.optionalAttrs (!pkgs.stdenv.isDarwin) {
            mnn-llvm-cov = craneLibLLvmTools.cargoLlvmCov (commonArgs // {inherit cargoArtifacts;});
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
              cargo-hakari
              cargo-deny
            ];
          };
        };
      }
    );
}
