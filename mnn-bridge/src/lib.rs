#[cfg(feature = "ndarray")]
pub mod ndarray;
#[cfg(feature = "ndarray_0_15")]
mod ndarray_0_15 {
    use ndarray_0_15 as ndarray;
    include!("ndarray.rs");
}
