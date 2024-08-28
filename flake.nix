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
              # emscripten = prev.emscripten.override {
              #   patches = [./patches/no-stack-ir.patch];
              # };
              # Need the master of binaryen since --no-stack-ir is not yet in the tagged release
              binaryen = prev.binaryen.overrideAttrs {
                src = pkgs.fetchFromGitHub {
                  owner = "WebAssembly";
                  repo = "binaryen";
                  rev = "d945aa489a1ad62c130e04ceea8492c7a728ab57";
                  hash = "sha256-mRm92P64J8fyODvwzU2bTjHQSVqr0rp0VvouGj2biVk=";
                };
              };
            })
          ];
        };
        inherit (pkgs) lib;

        # stableToolchain = pkgs.rust-bin.stable.latest.default;
        stableToolchainWithRustAnalyzer = pkgs.rust-bin.nightly.latest.default.override {
          extensions = ["rust-src" "rust-analyzer"];
          targets = [
            "wasm32-unknown-emscripten"
            "wasm32-unknown-unknown"
          ];
        };
        rustEmscriptenToolchainNightly = pkgs.rust-bin.nightly.latest.default.override {targets = ["wasm32-unknown-emscripten"];};
        rustWasmToolchainNightly = pkgs.rust-bin.nightly.latest.default.override {
          extensions = ["rust-src"];
          targets = ["wasm32-unknown-unknown"];
        };
        rustToolchain = pkgs.rust-bin.nightly.latest.default.override {extensions = ["rust-src"];};
        craneLibEmcc = (crane.mkLib pkgs).overrideToolchain rustEmscriptenToolchainNightly;
        craneLibWasm = (crane.mkLib pkgs).overrideToolchain rustWasmToolchainNightly;
        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
        src = ./.;
        MNN_SRC = pkgs.fetchFromGitHub {
          owner = "alibaba";
          repo = "MNN";
          rev = "e6042e5e00ba4f6398a5cd5a3615b9f62501438e";
          hash = "sha256-esHU+ociPi7qxficXU0dL+R5MXsblMocrNRgp79hWkk=";
        };
        emccArgs = {
          inherit src MNN_SRC;
          pname = "wasm-runner";
          EM_CONFIG = pkgs.writeText ".emscripten" (builtins.readFile "${pkgs.emscripten}/share/emscripten/.emscripten");
          configurePhase = ''
            runHook preConfigureHooks
            cp -r ${pkgs.emscripten}/share/emscripten/cache .emscripten_cache
            chmod -R u+w .emscripten_cache
            export EM_CACHE="$(realpath .emscripten_cache)"
            # export BINDGEN_EXTRA_CLANG_ARGS="--sysroot=$EM_CACHE/sysroot";
            # export BINDGEN_EXTRA_CLANG_ARGS="-I$EM_CACHE/sysroot/include";
            runHook postConfigure
          '';
          # BINDGEN_EXTRA_CLANG_ARGS = "-I${pkgs.emscripten}/share/emscripten/cache/sysroot/include";
          # BINDGEN_EXTRA_CLANG_ARGS = "--sysroot=${pkgs.emscripten}/share/emscripten/cache/sysroot";
          hardeningDisable = ["all"];
          cargoExtraArgs = "--package runner --target wasm32-unknown-emscripten";
          # buildPhaseCargoCommand = ''
          #   cargo build --package runner --release --target wasm32-unknown-emscripten
          # '';
          doCheck = false;
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          nativeBuildInputs = with pkgs; [
            emscripten
            cmake
          ];
        };
        wasmArgs =
          emccArgs
          // {
            RUSTFLAGS = "--Z wasm_c_abi=spec";
            cargoExtraArgs = "--package runner --target wasm32-unknown-unknown";
          };
        cargoArgs = {
          inherit src MNN_SRC;
          pname = "runner";
          cargoExtraArgs = "--package runner";
          doCheck = false;
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          BINDGEN_EXTRA_CLANG_ARGS = "-I${pkgs.llvmPackages.libclang.lib}/lib/clang/18/include";
          nativeBuildInputs = with pkgs; [
            cmake
            llvmPackages.libclang.lib
          ];
        };

        emscriptenArtifacts = craneLibEmcc.buildDepsOnly emccArgs;
        wasmArtifacts = craneLibWasm.buildDepsOnly wasmArgs;
      in {
        packages = rec {
          default = wasm-runner-emscripten;
          runner = craneLib.buildPackage cargoArgs;
          wasm-runner-unknown = craneLibWasm.buildPackage (wasmArgs
            // {
              cargoArtifacts = wasmArtifacts;
              # buildPhase = ''RUSTFLAGS="--Z wasm_c_abi=spec" cargo build --package runner --release --target wasm32-unknown-unknown'';
              buildPhase = ''
                HOME=./wasme RUSTFLAGS="--Z wasm_c_abi=spec --cfg=web_sys_unstable_apis" cargo build --package runner --release --target wasm32-unknown-unknown
                wasm-bindgen target/wasm32-unknown-unknown/release/runner.wasm --out-dir target/wasm32-unknown-unknown/release --target web || true
              '';
              nativeBuildInputs = wasmArgs.nativeBuildInputs ++ [pkgs.wasm-pack pkgs.wasm-bindgen-cli];
              installPhaseCommand = ''
                mkdir -p $out/bin
                # find target -type f -name '*.wasm' -exec cp {} $out/bin/ \;
                # find target -type f -name '*.js' -exec cp {} $out/bin/ \;
                # cp target/wasm32-unknown-unknown/release/{benchmark,runner}.{wasm,js} $out/bin/
                cp target/wasm32-unknown-unknown/release/*.{wasm,js,ts} $out/bin/
              '';
            });
          wasm-runner-emscripten = craneLibEmcc.buildPackage (emccArgs
            // {
              cargoArtifacts = emscriptenArtifacts;
              installPhaseCommand = ''
                mkdir -p $out/bin
                cp target/wasm32-unknown-emscripten/release/{benchmark,runner}.{wasm,js} $out/bin/
              '';
            });
          mnn-js = craneLibWasm.buildPackage (wasmArgs
            // {
              # cargoExtraArgs = "--package mnn-js --target wasm32-unknown-unknown";
              buildPhase = ''RUSTFLAGS="-Clink-arg=-fuse-ld=wasm-ld --Z wasm_c_abi=spec" cargo build --package mnn-js --release --target wasm32-unknown-unknown'';
              HOME = ".";
              nativeBuildInputs = wasmArgs.nativeBuildInputs ++ [pkgs.wasm-pack pkgs.wasm-bindgen-cli pkgs.lld];
              installPhaseCommand = ''
                mkdir -p $out/bin
                cp target/wasm32-unknown-unknown/release/mnn-js.{wasm,js} $out/bin/
              '';
            });
        };

        devShells = rec {
          default = wasm;
          wasm = pkgs.mkShell (emccArgs
            // {
              hardeningDisable = ["all"];
              packages = with pkgs; [
                mnn
                darwin.apple_sdk.frameworks.OpenCL
                llvmPackages.clang.cc
                rust-bindgen-unwrapped
                stableToolchainWithRustAnalyzer
                # rustup
                wasm-pack
                wasm-bindgen-cli
                binaryen
                nodejs_22
                miniserve
              ];
            });
        };
      }
    );
}
