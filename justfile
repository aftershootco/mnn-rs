VERSION:= "3.6.1"

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
    #!/usr/bin/env bash
    set -euo pipefail
    auth=()
    [[ -n "${GITHUB_TOKEN:-}" ]] && auth=(-H "Authorization: Bearer $GITHUB_TOKEN")
    release_json="$(curl -sSL --fail \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "${auth[@]}" \
        "https://api.github.com/repos/alibaba/MNN/releases/tags/{{version}}")"
    # Every `mnn_<version>_<suffix>.zip` asset, keyed by suffix (the backend list in the
    # suffix changes between releases, so mnn-sys looks the names up by platform prefix).
    jq --arg version "{{version}}" '{
        version: $version,
        checksums: ([ .assets[]
            | select(.name | startswith("mnn_" + $version + "_") and endswith(".zip"))
            | { key: (.name | ltrimstr("mnn_" + $version + "_") | rtrimstr(".zip")), value: .digest }
        ] | from_entries)
    }' <<<"$release_json" > mnn-sys/build/checksums.json
    jq -r '.checksums | to_entries[] | "  \(.key) -> \(.value)"' mnn-sys/build/checksums.json >&2
    echo "Wrote mnn-sys/build/checksums.json" >&2


download version=VERSION:
    mkdir -p downloads 
    curl -L -H "Accept: application/vnd.github+json" -H "-X-GitHub-Api-Version: 2022-11-28" "https://api.github.com/repos/alibaba/MNN/releases/tags/{{version}}" | jq -Sr '.assets[].browser_download_url' | xargs -n 1 curl -L -O --output-dir downloads/


