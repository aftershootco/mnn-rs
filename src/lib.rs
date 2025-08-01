#![deny(missing_docs)]
//!
//! Ergonomic rust bindings for [MNN](https://github.com/alibaba/MNN)
//!
//! The main data structures used are [`Tensor`] and [`Interpreter`].
//! [Interpreter] should be thread safe and can be used to run multiple sessions concurrently.
//! [Send] / [Sync] is not implemented for Interpreter yet since we don't know how it will be used.
//!
//! ![Codecov](https://img.shields.io/codecov/c/github/aftershootco/mnn-rs?link=https%3A%2F%2Fapp.codecov.io%2Fgithub%2Faftershootco%2Fmnn-rs)
//! ![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/aftershootco/mnn-rs/build.yaml?link=https%3A%2F%2Fgithub.com%2Faftershootco%2Fmnn-rs%2Factions%2Fworkflows%2Fbuild.yaml)
//! # Example
//! ```rust,no_run
//! use mnn::*;
//! let mut interpreter = Interpreter::from_bytes([0;100]).unwrap();
//! let mut sc = ScheduleConfig::new();
//! let session = interpreter.create_session(sc).unwrap();
//! let mut input = interpreter.input::<f32>(&session, "input").unwrap();
//! let mut tensor = input.create_host_tensor_from_device(false);
//! tensor.host_mut().fill(1.0f32);
//! input.copy_from_host_tensor(tensor.view()).unwrap();
//! interpreter.run_session(&session).unwrap();
//! let output = interpreter.output::<u8>(&session, "output").unwrap();
//! let mut output_tensor = output.create_host_tensor_from_device(true);
//! std::fs::write("output.bin", output_tensor.host().to_vec()).unwrap();
//! ```
//! **NOTE:**  The library is still in development and the API is subject to change.
//!
//! ## Features
//! - `metal`: Enable mnn Metal backend
//! - `coreml`: Enable mnn CoreML backend
//! - `vulkan`: Enable mnn Vulkan backend (unimplemented from rust wrapper)
//! - `opencl`: Enable mnn OpenCL backend
//! - `opengl`: Enable mnn OpenGL backend (unimplemented from rust wrapper)
//! - `openmp`: Enable mnn Openmp ( disable the mnn-threadpool feature to enable this)
//! - `mnn-threadpool`: Enable mnn threadpool ( enabled by default can't be used with openmp)
//! - `sync`: Enable sync api
//! - `profile`: Enable profiling ( emits some profiling tracing events )
//! - `tracing`: Enable tracing ( emits some tracing events )
//! - `crt_static`: Link statically to the C runtime on windows (noop on other platforms)
//! ## License
//! This links to the MNN library which is licensed under the Apache License 2.0.
//! The rust bindings are licensed under the same Apache License 2.0.
//!
//! ## Building
//! The flake.nix provides a nix-shell with all the dependencies required to build the library.
//! If not using nix you'll need to clone the git submodule to get the MNN source code in mnn-sys/vendor first
//! Or you can export the MNN_SRC environment variable to point to the MNN source code.
//!
//! ## Compatibility Chart for current crate
//! | MNN Backend | Compiles | Works |
//! | ----------- | -------- | ----- |
//! | CPU         | ✅       | ✅    |
//! | OpenCL      | ✅       | ✅    |
//! | Metal       | ✅       | ✅    |
//! | CoreML      | ✅       | 🚸    |
//! | OpenGL      | ❌       | ❌    |
//! | Vulkan      | ❌       | ❌    |
//!
//! - ✅ - Works
//! - 🚸 - Some models work
//! - ❌ - Doesn't work

/// Re-export of whole mnn-sys
pub mod ffi {
    pub use mnn_sys::*;
}

mod profile;

pub mod backend;
/// Error handling
pub mod error;
/// MNN::Interpreter related items
pub mod interpreter;
/// Schedule configuration
pub mod schedule;
/// MNN::Session related items
pub mod session;
/// MNN::Tensor related items
pub mod tensor;

pub use backend::*;
pub use error::*;
pub use interpreter::*;
pub use schedule::*;
pub use session::*;
pub use tensor::*;

pub use ffi::HalideType;
pub use ffi::MapType;

/// Re-export of commonly used items
pub mod prelude {
    pub use crate::error::*;
    pub(crate) use crate::profile::profile;
    pub use crate::tensor::{
        Device, Host, Owned, TensorMachine, TensorType, TensorView, TensorViewMut, View,
    };
    pub use core::marker::PhantomData;
    pub use error_stack::{Report, ResultExt};
    pub use libc::*;
    pub use mnn_sys::{HalideType, MapType};
}
