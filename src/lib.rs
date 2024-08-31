//! # MNN
//!
//! Ergonomic rust bindings for MNN
//!
//! The main data structures used are [`Tensor`] and [`Interpreter`].   
//! [Interpreter] should be thread safe and can be used to run multiple sessions concurrently.  
//! [Send] / [Sync] is not implemented for Interpreter yet since we don't know how it will be used.  
//!
//! # Example  
//! ```rust,no_run
//! use mnn::*;
//! let mut interpreter = Interpreter::from_bytes([0;100]).unwrap();
//! let mut sc = ScheduleConfig::new();
//! let session = interpreter.create_session(&mut sc).unwrap();
//! let mut input = interpreter.input::<f32>(&session, "input").unwrap();
//! let mut tensor = input.create_host_tensor_from_device(false);
//! tensor.host_mut().fill(1.0f32);
//! input.copy_from_host_tensor(&tensor).unwrap();
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
//! - `parse`: Enable parsing of some enums from string

pub mod ffi {
    pub use mnn_sys::*;
}

pub mod backend;
pub mod error;
pub mod interpreter;
pub mod schedule;
pub mod session;
pub mod tensor;

pub use backend::*;
pub use error::*;
pub use interpreter::*;
pub use schedule::*;
pub use session::*;
pub use tensor::*;

pub mod sync;

pub mod prelude {
    pub use crate::error::*;
    pub use core::marker::PhantomData;
    pub use error_stack::{Report, ResultExt};
    pub use libc::*;
}
