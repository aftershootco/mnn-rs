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
      url = "github:alibaba/MNN/3.2.0";
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
                src = mnn-src;
                buildConverter = true;
                enableMetal = true;
                enableOpencl = true;
              };
            })
          ];
        };
        inherit (pkgs) lib;

        version = "latest";

        rustToolchain = pkgs.rust-bin.stable.${version}.default;
        rustToolchainWithLLvmTools = pkgs.rust-bin.stable.${version}.default.override {
          extensions = ["rust-src" "llvm-tools"];
        };
        rustToolchainWithRustAnalyzer = pkgs.rust-bin.stable.${version}.default.override ({
            extensions = ["rust-docs" "rust-src" "rust-analyzer"];
          }
          // (lib.optionalAttrs pkgs.stdenv.isDarwin {
            targets = ["aarch64-apple-darwin" "x86_64-apple-darwin"];
          }));
        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
        craneLibLLvmTools = (crane.mkLib pkgs).overrideToolchain rustToolchainWithLLvmTools;

        src = lib.sources.sourceFilesBySuffices ./. [".rs" ".toml" ".patch" ".mnn" ".h" ".cpp" ".svg" "lock"];
        MNN_SRC = pkgs.applyPatches {
          name = "mnn-src";
          src = mnn-src;
          patches = [./mnn-sys/patches/mnn-tracing.patch];
        };
        commonArgs = {
          inherit src MNN_SRC;
          pname = "mnn";
          doCheck = false;
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          nativeBuildInputs = with pkgs; [
            cmake
            llvmPackages.libclang.lib
            clang
            pkg-config
          ];
          buildInputs = with pkgs;
            []
            ++ (lib.optionals pkgs.stdenv.isLinux [
              ocl-icd
              opencl-headers
            ])
            ++ (lib.optionals pkgs.stdenv.isDarwin [
              apple-sdk_13
            ]);
        };
        cargoArtifacts = craneLib.buildPackage commonArgs;
      in rec {
        checks =
          {
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
            # Audit dependencies
            mnn-audit =
              craneLib.cargoAudit.override {
                cargo-audit = pkgs.cargo-audit;
              } {
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
            # mnn-asan = let
            #   rustPlatform = pkgs.makeRustPlatform {
            #     cargo = nightlyToolchain;
            #     rustc = nightlyToolchain;
            #   };
            # in
            #   rustPlatform.buildRustPackage (
            #     commonArgs
            #     // {
            #       inherit src;
            #       name = "mnn-leaks";
            #       cargoLock = {
            #         lockFile = ./Cargo.lock;
            #         outputHashes = {
            #           "cmake-0.1.50" = "sha256-GM2D7dpb2i2S6qYVM4HYk5B40TwKCmGQnUPfXksyf0M=";
            #         };
            #       };
            #
            #       buildPhase = ''
            #         cargo test --target aarch64-apple-darwin
            #       '';
            #       RUSTFLAGS = "-Zsanitizer=address";
            #       ASAN_OPTIONS = "detect_leaks=1";
            #       # MNN_COMPILE = "NO";
            #       # MNN_LIB_DIR = "${pkgs.mnn}/lib";
            #     }
            #   );
          }
          // lib.optionalAttrs (!pkgs.stdenv.isDarwin) {
            mnn-llvm-cov = craneLibLLvmTools.cargoLlvmCov (commonArgs // {inherit cargoArtifacts;});
          };
        packages = rec {
          mnn = craneLib.buildPackage (commonArgs
            // {
              inherit cargoArtifacts;
            });
          inspect = craneLib.buildPackage (commonArgs
            // {
              inherit cargoArtifacts;
              pname = "inspect";
              cargoExtraArgs =
                "--example inspect"
                + (
                  lib.optionalString pkgs.stdenv.isDarwin " --features opencl,metal,coreml" # + lib.optionalString pkgs.stdenv.isAarch64 ",metal,coreml"
                );
            });
          bencher = craneLib.buildPackage (commonArgs
            // {
              inherit cargoArtifacts;
              pname = "bencher";
              cargoExtraArgs = "--package bencher";
            });
          default = mnn;
        };

        devShells = {
          default = pkgs.mkShell (commonArgs
            // {
              MNN_SRC = null;
              LLDB_DEBUGSERVER_PATH = "/Applications/Xcode.app/Contents/SharedFrameworks/LLDB.framework/Versions/A/Resources/debugserver";
              packages = with pkgs;
                [
                  cargo-audit
                  cargo-deny
                  cargo-hakari
                  cargo-nextest
                  cargo-semver-checks
                  clang
                  git
                  git-lfs
                  llvm
                  llvmPackages.lldb
                  nushell
                  rust-bindgen
                  google-cloud-sdk
                  rustToolchainWithRustAnalyzer
                  mnn
                ]
                ++ (
                  lib.optionals pkgs.stdenv.isLinux [
                    cargo-llvm-cov
                  ]
                );
            });
        };
      }
    )
    // {
      githubActions = nix-github-actions.lib.mkGithubMatrix {
        checks = nixpkgs.lib.getAttrs ["x86_64-linux" "aarch64-darwin"] self.checks;
      };
    };
}
