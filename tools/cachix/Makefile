.PHONY: checks push push-all

all: push-all

checks: 
	nix flake check

push:
	cachix watch-exec mnn-rs -- nix flake check

push-all:
	cachix watch-exec mnn-rs -- nix flake check --system x86_64-linux
	cachix watch-exec mnn-rs -- nix flake check --system aarch64-darwin

	
