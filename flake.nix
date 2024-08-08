{
  description = "A simple rust flake using rust-overlay and craneLib";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs";
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
            mnn-overlay.overlays.${system}.default
            (final: prev: {
              cargo-with = let
                pname = "cargo-with";
                version = "0.3.2";
                src = final.fetchCrate {
                  inherit pname version;
                  hash = "sha256-USBrtvN+3MZTeBPYSwxnZ3m6kCoBwuhU7NSBX5kwYSQ=";
                };
              in
                final.rustPlatform.buildRustPackage rec {
                  inherit pname version src;
                  cargoLock = {lockFile = "${src}/Cargo.lock";};
                  doCheck = false;
                };
            })
          ];
        };
        inherit (pkgs) lib;

        stableToolchain = pkgs.rust-bin.stable.latest.default;
        stableToolchainWithRustAnalyzer = pkgs.rust-bin.stable.latest.default.override {
          extensions = ["rust-src" "rust-analyzer"];
          # Extra targets if required
          targets = [
            #   "x86_64-unknown-linux-gnu"
            # "x86_64-unknown-linux-musl"
            "wasm32-unknown-emscripten"
            #   "x86_64-apple-darwin"
            #   "aarch64-apple-darwin"
          ];
        };
        craneLib = (crane.mkLib pkgs).overrideToolchain stableToolchain;
        src = ./.;
        commonArgs = {
          inherit src;
          cargoExtraArgs = "--package runner --target wasm32-unknown-emscripten";
          buildInputs = with pkgs;
            [
              # (mnn.override {
              #   # enableMetal = true;
              #   enableVulkan = true;
              #   buildConverter = true;
              # })
            ]
            ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
              libiconv
              pkgs.darwin.apple_sdk.frameworks.Metal
              pkgs.darwin.apple_sdk.frameworks.OpenCL
              pkgs.darwin.apple_sdk.frameworks.CoreML
              pkgs.darwin.apple_sdk.frameworks.CoreVideo
            ];

          nativeBuildInputs = with pkgs;
            [
              cmake
              pkg-config
              emscripten
              rustPlatform.bindgenHook
            ]
            ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
              xcbuild
            ];
        };
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;
        mnn-runner = craneLib.buildPackage (commonArgs
          // {
            inherit cargoArtifacts;
          });
      in {
        checks = {
          mnn-runner-clippy = craneLib.cargoClippy (commonArgs
            // {
              inherit cargoArtifacts;
              cargoClippyExtraArgs = "--all-targets -- --deny warnings";
            });
          mnn-runner-fmt = craneLib.cargoFmt {
            inherit src;
          };
          mnn-runner-nextest = craneLib.cargoNextest (commonArgs
            // {
              inherit cargoArtifacts;
              partitions = 1;
              partitionType = "count";
            });
        };
        packages.default = mnn-runner;

        devShells = rec {
          default = wasm;
          wasm = pkgs.mkShell {
            hardeningDisable = ["all"];
            buildInputs = with pkgs; [opencl-headers];
            packages = with pkgs; [
              llvmPackages.clang.cc
              rust-bindgen-unwrapped
              cmake
              mnn
              libiconv
            ];
          };
          rust = (craneLib.overrideToolchain stableToolchainWithRustAnalyzer).devShell (commonArgs
            // {
              hardeningDisable = ["all"];
              packages = with pkgs; [
                lldb
                cargo-with
                cargo-expand
                delta
              ];
            });
        };
      }
    );
}
