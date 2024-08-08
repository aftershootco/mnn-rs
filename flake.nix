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
        };
        rustEmscriptenToolchainNightly = pkgs.rust-bin.nightly.latest.default.override {targets = ["wasm32-unknown-emscripten"];};
        craneLibEmcc = (crane.mkLib pkgs).overrideToolchain rustEmscriptenToolchainNightly;
        # src = ./.;
        # commonArgs = {
        #   inherit src;
        #   cargoExtraArgs = "--package runner --target wasm32-unknown-emscripten";
        #   buildInputs = with pkgs;
        #     []
        #     ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
        #       libiconv
        #       pkgs.darwin.apple_sdk.frameworks.Metal
        #       pkgs.darwin.apple_sdk.frameworks.OpenCL
        #       pkgs.darwin.apple_sdk.frameworks.CoreML
        #       pkgs.darwin.apple_sdk.frameworks.CoreVideo
        #     ];
        #
        #   nativeBuildInputs = with pkgs;
        #     [
        #       cmake
        #       pkg-config
        #       emscripten
        #       rustPlatform.bindgenHook
        #     ]
        #     ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
        #       xcbuild
        #     ];
        # };
        # cargoArtifacts = craneLibEmcc.buildDepsOnly commonArgs;
        # mnn-runner = craneLibEmcc.buildPackage (commonArgs
        #   // {
        #     inherit cargoArtifacts;
        #   });
        emccArgs = {
          pname = "wasm-runner";
          version = "0.1.0";
          src = ./.;
          EM_CONFIG = pkgs.writeText ".emscripten" ((builtins.readFile "${pkgs.emscripten}/share/emscripten/.emscripten")
            + ''
            '');
          EM_CACHE = ".emscripten_cache";
          configurePhase = ''
            runHook preConfigureHooks
            cp -r ${pkgs.emscripten}/share/emscripten/cache .emscripten_cache
            chmod -R u+w .emscripten_cache
            runHook postConfigure
          '';
          # CACHE = '${pkgs.emscripten}/share/emscripten/cache'
          MNN_SRC = pkgs.fetchFromGitHub {
            "owner" = "alibaba";
            "repo" = "MNN";
            "rev" = "e6042e5e00ba4f6398a5cd5a3615b9f62501438e";
            hash = "sha256-esHU+ociPi7qxficXU0dL+R5MXsblMocrNRgp79hWkk=";
          };
          hardeningDisable = ["all"];
          buildPhase = ''
            cargo build --release --target wasm32-unknown-emscripten
          '';
          nativeBuildInputs = with pkgs; [
            emscripten
            cmake
          ];
        };
        emscriptenArtifacts = craneLibEmcc.buildDepsOnly emccArgs;
      in {
        packages = {
          # default = mnn-runner;
          wasm-runner-artifacts = emscriptenArtifacts;
          wasm-runner = craneLibEmcc.buildPackage (emccArgs // {cargoArtifacts = emscriptenArtifacts;});
        };
        # checks = {
        #   mnn-runner-clippy = craneLibEmcc.cargoClippy (commonArgs
        #     // {
        #       inherit cargoArtifacts;
        #       cargoClippyExtraArgs = "--all-targets -- --deny warnings";
        #     });
        #   mnn-runner-fmt = craneLibEmcc.cargoFmt {
        #     inherit src;
        #   };
        #   mnn-runner-nextest = craneLibEmcc.cargoNextest (commonArgs
        #     // {
        #       inherit cargoArtifacts;
        #       partitions = 1;
        #       partitionType = "count";
        #     });
        # };

        devShells = rec {
          default = wasm;
          wasm = pkgs.mkShell {
            hardeningDisable = ["all"];
            buildInputs = with pkgs; [opencl-headers];
            packages = with pkgs; [
              llvmPackages.clang.cc
              rust-bindgen-unwrapped
              cmake
              # mnn
              # libiconv
              emscripten
              (pkgs.rust-bin.stable.latest.default.override {targets = ["wasm32-unknown-emscripten"];})
            ];
            LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          };
          # rust = (craneLibEmcc.overrideToolchain stableToolchainWithRustAnalyzer).devShell (commonArgs
          #   // {
          #     hardeningDisable = ["all"];
          #     packages = with pkgs; [
          #       lldb
          #       cargo-with
          #       cargo-expand
          #       delta
          #     ];
          #   });
        };
      }
    );
}
