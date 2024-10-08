{
  description = "A simple rust flake using rust-overlay and craneLib";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";
    nix-github-actions = {
      url = "github:nix-community/nix-github-actions";
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
    advisory-db = {
      url = "github:rustsec/advisory-db";
      flake = false;
    };
    mnn-src = {
      url = "github:alibaba/MNN/2.9.5";
      flake = false;
    };
  };

  outputs = {
    self,
    crane,
    flake-utils,
    nixpkgs,
    rust-overlay,
    mnn-overlay,
    advisory-db,
    nix-github-actions,
    mnn-src,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            rust-overlay.overlays.default
            (final: prev: {
              mnn = mnn-overlay.packages.${system}.mnn.override {
                buildConverter = true;
                enableVulkan = false;
              };
            })
          ];
        };
        inherit (pkgs) lib;

        stableToolchain = pkgs.rust-bin.stable.latest.default;
        nightlyToolchain = pkgs.rust-bin.nightly.latest.default.override {
          extensions = ["rust-src"];
        };
        stableToolchainWithLLvmTools = pkgs.rust-bin.stable.latest.default.override {
          extensions = ["rust-src" "llvm-tools"];
        };
        stableToolchainWithRustAnalyzer = pkgs.rust-bin.stable.latest.default.override {
          extensions = ["rust-src" "rust-analyzer"];
        };
        craneLib = (crane.mkLib pkgs).overrideToolchain stableToolchain;
        craneLibLLvmTools = (crane.mkLib pkgs).overrideToolchain stableToolchainWithLLvmTools;

        mnnFilters = path: type: (craneLib.filterCargoSources path type) || (lib.hasSuffix ".patch" path || lib.hasSuffix ".mnn" path || lib.hasSuffix ".h" path || lib.hasSuffix ".cpp" path || lib.hasSuffix ".svg" path);
        src = lib.cleanSourceWith {
          filter = mnnFilters;
          src = ./.;
        };
        MNN_SRC = mnn-src;
        commonArgs =
          {
            inherit src MNN_SRC;
            pname = "mnn";
            doCheck = false;
            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
            nativeBuildInputs = with pkgs; [
              cmake
              llvmPackages.libclang.lib
            ];
            buildInputs = with pkgs;
              []
              ++ (lib.optionals pkgs.stdenv.isDarwin [
                darwin.apple_sdk.frameworks.OpenCL
                darwin.apple_sdk.frameworks.OpenGL
                darwin.apple_sdk.frameworks.CoreML
                darwin.apple_sdk.frameworks.Metal
              ]);
          }
          // (lib.optionalAttrs pkgs.stdenv.isLinux {
            BINDGEN_EXTRA_CLANG_ARGS = "-I${pkgs.llvmPackages.libclang.lib}/lib/clang/18/include";
          });
        cargoArtifacts = craneLib.buildPackage commonArgs;
      in {
        checks = {
          mnn-clippy = craneLib.cargoClippy (commonArgs
            // {
              inherit cargoArtifacts;
              cargoClippyExtraArgs = "--all-targets -- --deny warnings";
            });
          mnn-docs = craneLib.cargoDoc (commonArgs
            // {
              inherit cargoArtifacts;
              cargoDocExtraArgs = "-p mnn -p mnn-sys";
            });
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
          mnn-nextest = craneLib.cargoNextest (commonArgs
            // {
              inherit cargoArtifacts;
              partitions = 1;
              partitionType = "count";
            });
          mnn-sys-clippy = craneLib.cargoClippy (commonArgs
            // {
              inherit cargoArtifacts;
              cargoClippyExtraArgs = "-p mnn-sys --all-targets -- --deny warnings";
            });
          mnn-sys-nextest = craneLib.cargoNextest (commonArgs
            // {
              inherit cargoArtifacts;
              partitions = 1;
              partitionType = "count";
              cargoExtraArgs = "-p mnn-sys";
            });
          mnn-asan = (craneLib.overrideToolchain nightlyToolchain).cargoNextest (commonArgs
            // {
              inherit cargoArtifacts;
              partitions = 1;
              partitionType = "count";
              RUSTFLAGS = "-Zsanitizer=address -Zbuild-std";
            });
        };
        packages =
          rec {
            mnn = craneLib.buildPackage (commonArgs
              // {
                inherit cargoArtifacts;
              });
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
            packages = with pkgs;
              [
                mnn
                stableToolchainWithRustAnalyzer
                cargo-nextest
                cargo-hakari
                cargo-deny
                cargo-semver-checks
                rust-bindgen
              ]
              ++ (lib.optionals pkgs.stdenv.isDarwin [
                darwin.apple_sdk.frameworks.OpenCL
                darwin.apple_sdk.frameworks.CoreML
                darwin.apple_sdk.frameworks.Metal
              ]);
          };
        };
      }
    )
    // {
      githubActions = nix-github-actions.lib.mkGithubMatrix {
        checks = nixpkgs.lib.getAttrs ["x86_64-linux"] self.checks;
      };
    };
}
