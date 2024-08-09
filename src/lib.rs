pub mod ffi {
    pub use mnn_sys::*;
}

// pub mod tensor;
// pub use interpreter::Interpreter;
pub mod interpreter;
pub mod session;
pub mod tensor;
pub use interpreter::*;
pub use session::*;
pub use tensor::*;
pub mod backend;
pub use backend::*;
pub mod error;
pub use error::*;
pub mod schedule;
pub use schedule::*;
pub mod prelude {
    pub use crate::error::*;
    pub use core::marker::PhantomData;
    pub use error_stack::{Report, ResultExt};
    pub use core::ffi::*;
}

#[cfg(feature = "clap")]
pub mod utils;

