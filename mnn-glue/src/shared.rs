#[cxx::bridge(namespace = "MNN")]
mod shared {
    #[derive(Debug, Clone)]
    pub struct TensorShape {
        pub dims: Vec<i32>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum HalideTypes {
        halide_float,
        halide_double,
        halide_bool,
        halide_uint8_t,
        halide_uint16_t,
        halide_uint32_t,
        halide_uint64_t,
        halide_int8_t,
        halide_int16_t,
        halide_int32_t,
        halide_int64_t,
    }
}

pub use shared::*;
