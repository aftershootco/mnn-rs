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
  };

  outputs = {
    crane,
    flake-utils,
    nixpkgs,
    rust-overlay,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [rust-overlay.overlays.default];
        };
        inherit (pkgs) lib;

        stableToolchain = pkgs.rust-bin.stable.latest.default;
        stableToolchainWithRustAnalyzer = pkgs.rust-bin.stable.latest.default.override {
          extensions = ["rust-src" "rust-analyzer"];
          # Extra targets if required
          # targets = [
          #   "x86_64-unknown-linux-gnu"
          #   "x86_64-unknown-linux-musl"
          #   "x86_64-apple-darwin"
          #   "aarch64-apple-darwin"
          # ];
        };
        craneLib = (crane.mkLib pkgs).overrideToolchain stableToolchain;
        src = craneLib.cleanCargoSource (craneLib.path ./.);
        commonArgs = {
          inherit src;
          buildInputs = with pkgs;
            []
            ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
              libiconv
              # pkgs.darwin.apple_sdk.frameworks.CoreServices
              # pkgs.darwin.apple_sdk.frameworks.Security
              # pkgs.darwin.apple_sdk.frameworks.SystemConfiguration
              # pkgs.darwin.apple_sdk.frameworks.Foundation
              pkgs.darwin.apple_sdk.frameworks.Metal
            ]; # Inputs required for the TARGET system

          nativeBuildInputs = with pkgs; [
            # often required for c/c++ libs
            pkg-config
            rustPlatform.bindgenHook
          ]; # Intputs required for the HOST system
          # This is often requird for any ffi based packages that use bindgen
          # LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          # For using pkg-config that many libraries require
          # PKG_CONFIG_PATH = lib.makeSearchPath "lib/pkgconfig" (with pkgs;[ openssl.dev zlib.dev ]);
        };
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;
        hello = craneLib.buildPackage (commonArgs
          // {
            inherit cargoArtifacts;
          });
      in {
        checks = {
          hello-clippy = craneLib.cargoClippy (commonArgs
            // {
              inherit cargoArtifacts;
              cargoClippyExtraArgs = "--all-targets -- --deny warnings";
            });
          hello-fmt = craneLib.cargoFmt {
            inherit src;
          };
          hello-nextest = craneLib.cargoNextest (commonArgs
            // {
              inherit cargoArtifacts;
              partitions = 1;
              partitionType = "count";
            });
        };
        packages.hello = hello;

        devShells.default = (craneLib.overrideToolchain stableToolchainWithRustAnalyzer).devShell (commonArgs
          // {
            packages = with pkgs; [
              cargo-nextest
              cargo-criterion
              cargo-expand
            ];
          });
      }
    );
}
