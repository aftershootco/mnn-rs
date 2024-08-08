# MNN-sys

This contains the C and rust bindings to the MNN libraries main functions

For emscripten

If you want to use wasm32-unknown-unknown (the main target of the rust-wasm ecosystem) then you'll need to compile the C++ library from emscripten 
and link to it using the -Zwasm-c-abi
The speed is pretty slow however.

For wasm32-unknown-emscripten targets it should work out of the box.
