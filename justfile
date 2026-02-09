push-cachix:
    nix eval .#checks.aarch64-darwin --json | jq -r '.[]' | cachix push mnn-rs
    nix eval .#checks.x86_64-linux --json | jq -r '.[]' | cachix push mnn-rs
publish:
    cargo publish --package mnn-sys
    cargo publish --package mnn

package:
    cargo package --package mnn-sys
    cargo package --package mnn

version name:
    cargo metadata --no-deps --format-version 1 | jq -r '.packages.[] | select(.name == "{{name}}") | .version'
