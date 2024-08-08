wasm:
    nix develop .#wasm
rust:
    nix develop .#rust

wbench:
    nix develop .#wasm --command cargo build "--release" "--target=wasm32-unknown-emscripten" "--example" wasm-bench "--features" openmp,error-report 
    cp $CARGO_TARGET_DIR/wasm32-unknown-emscripten/release/examples/{wasm_bench.wasm,wasm_bench.js} ./

