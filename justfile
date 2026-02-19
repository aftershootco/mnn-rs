VERSION:= "3.4.0"

push-cachix:
    nix flake check
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

checksums version=VERSION:
    curl -L -H "Accept: application/vnd.github+json" -H "-X-GitHub-Api-Version: 2022-11-28" "https://api.github.com/repos/alibaba/MNN/releases/tags/{{version}}" | jq -r '.assets[] | { name: .name, digest: .digest }'
    
download version=VERSION:
    mkdir -p downloads 
    curl -L -H "Accept: application/vnd.github+json" -H "-X-GitHub-Api-Version: 2022-11-28" "https://api.github.com/repos/alibaba/MNN/releases/tags/{{version}}" | jq -Sr '.assets[].browser_download_url' | xargs -n 1 curl -L -O --output-dir downloads/


