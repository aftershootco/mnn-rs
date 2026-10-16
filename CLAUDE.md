# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rust bindings for [alibaba/MNN](https://github.com/alibaba/MNN), a lightweight deep neural network inference engine. The crate wraps MNN's C++ library through a handwritten C bridge layer (`mnn-sys/mnn_c/`) with auto-generated Rust FFI bindings via `bindgen`.

## Build & Development

### Prerequisites
- **Nix (recommended):** `flake.nix` provides the complete dev environment. Just use `nix develop` or direnv.
- **Without Nix:** Clone git submodules first (`git submodule update --init --recursive`) or set `MNN_SRC` env var to point to MNN source. Requires CMake and clang/LLVM.
- **Windows:** Only compiles in `--release` mode due to msvcrt/MTd linking issues.

### Common Commands
```sh
cargo build                          # Build (CPU-only, default)
cargo build --features metal         # Build with Metal backend
cargo test                           # Run tests
cargo test --test basic              # Run a specific test file
cargo test test_name                 # Run a single test by name
cargo bench                          # Run benchmarks (uses divan)
```

### Features
- **Backends:** `metal`, `coreml`, `opencl`, `vulkan` (unimplemented), `opengl` (unimplemented)
- **Threading:** `mnn-threadpool` (default) and `openmp` are mutually exclusive
- **Other:** `tracing`, `profile`, `serde`, `crt_static`, `download`

### Backend Compatibility
| Backend | Status |
|---------|--------|
| CPU     | Works  |
| Metal   | Works  |
| OpenCL  | Works  |
| CoreML  | Partial (some models) |
| Vulkan/OpenGL | Not implemented |

## Workspace Structure

- **`mnn`** (root) — Main crate with safe Rust API
- **`mnn-sys`** — FFI bindings; `build.rs` compiles MNN C++ via cmake, generates bindings via bindgen. Build logic is modularized in `mnn-sys/build/` (`bindgen.rs`, `compile.rs`, `download.rs`, `options.rs`). Vendor source lives in `mnn-sys/vendor/` (git submodule).
- **`mnn-bridge`** — ndarray integration (supports versions 0.15–0.17)
- **`mnn-sync`** — Sync/async primitives for running inference sessions with `flume` channels
- **`tools/bencher`** — Benchmarking tool

## Architecture

### Type-Safe Tensor System

The core abstraction is `Tensor<S, M>` parameterized on two type-level dimensions:

1. **Ownership (`S`: TensorType):** `Owned<H>`, `View<&H>`, `View<&mut H>`
2. **Location (`M`: TensorMachine):** `Host` (CPU memory), `Device` (GPU memory)

Type aliases: `TensorOwned<H, M>`, `TensorView<'t, H, M>`, `TensorViewMut<'t, H, M>`

The element type `H` must implement `HalideType` (defined in `mnn-sys`): f32, f64, i8–i64, u8–u64, bool.

Key patterns:
- **Sealed traits** (`seal::Sealed`) prevent external implementations of `TensorType`, `TensorMachine`, `MutableTensorType`
- **`TensorRef<H, M>`** is an unsized reference type (like `[T]`) with `Deref`/`DerefMut` from `Tensor`
- Certain operations are restricted by type: `try_host`/`try_host_mut` only on `Host` tensors, `borrowed`/`borrowed_mut` only on `Host` `View` tensors

### Inference Pipeline

`Interpreter` → loads model (from file/bytes) → creates `Session` (with `ScheduleConfig` + `BackendConfig`) → get input/output tensors by name → fill inputs → `run_session` → read outputs.

`Session` is lifetime-bound to its `Interpreter` (`'i`).

### Error Handling

`MNNError` wraps `error_stack::Report<ErrorKind>`. Internal macros `ensure!` and `error!` (in `src/error.rs`) provide ergonomic error construction with context attachment.

## CI

Uses Nix-based GitHub Actions on macOS and Linux with Cachix binary caching. Code coverage via `llvm-cov` uploaded to Codecov.
