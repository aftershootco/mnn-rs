//! Conversions between [`mnn`] tensors and `ndarray` arrays.
//!
//! Every supported `ndarray` major version lives in its own module, gated behind the feature
//! of the same name: `ndarray_0_15`, `ndarray_0_16` and `ndarray_0_17`. The `ndarray`
//! feature (enabled by default) is an alias for the newest version and the `ndarray` module
//! is an alias for its module. Enabling several versions at the same time is supported, so
//! feature unification across a workspace can never break the build.

#[cfg(feature = "ndarray_0_17")]
pub mod ndarray_0_17 {
    use ndarray_0_17 as ndarray;
    include!("ndarray.rs");
}

#[cfg(feature = "ndarray_0_17")]
pub use self::ndarray_0_17 as ndarray;

#[cfg(feature = "ndarray_0_16")]
pub mod ndarray_0_16 {
    use ndarray_0_16 as ndarray;
    include!("ndarray.rs");
}

#[cfg(feature = "ndarray_0_15")]
pub mod ndarray_0_15 {
    use ndarray_0_15 as ndarray;
    include!("ndarray.rs");
}
