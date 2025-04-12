# MNN Rust Bindings

Rust bindings for the [MNN](https://github.com/alibaba/MNN) neural network inference engine.

## Installation

### Option 1: Automatic Download (Default)

This library will attempt to automatically download MNN source code when needed, but due to potential repository structure changes, it may not always work with the latest versions.

### Option 2: Git Submodule (Recommended)

The most reliable method is to use git submodules:

```bash
git submodule update --init --recursive
```

### Option 3: Manual Download

You can also manually download MNN and specify its location:

1. Clone MNN repository: `git clone https://github.com/alibaba/MNN.git /path/to/mnn`
2. Set environment variable: `export MNN_SRC=/path/to/mnn`
3. Build your project: `cargo build`


## Environment Variables

| Variable | Description |
|----------|-------------|
| `MNN_SRC` | Path to MNN source code |
| `MNN_VERSION` | When downloading, specifies version tag (default: "3.0.5") |
| `MNN_COMPILE` | Set to "0" to skip compilation (requires `MNN_LIB_DIR`) |
| `MNN_LIB_DIR` | Path to pre-built MNN libraries |
| `MNN_SYSTEM` | Set to "1" to use system-installed MNN libraries |
| `MNN_FORCE_DOWNLOAD` | Set to "1" to force re-download of MNN source |

## Using System Libraries

To use system-installed MNN libraries:
```bash
MNN_SYSTEM=1 cargo build
```

## Features

- `metal` - Enable Metal backend (Apple platforms)
- `coreml` - Enable CoreML backend (Apple platforms)
- `vulkan` - Enable Vulkan backend
- `opencl` - Enable OpenCL backend
- `opengl` - Enable OpenGL backend
- `crt_static` - Use static CRT on Windows
- `openmp` - Enable OpenMP (disables MNN threadpool)
- `mnn-threadpool` - Use MNN's threadpool (default)
- `tracing` - Enable tracing support
- `profile` - Enable profiling support
- `serde` - Enable serialization/deserialization support

## Examples

Check the `examples` directory for usage examples.



