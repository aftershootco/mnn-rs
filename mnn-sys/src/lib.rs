use std::ffi::CStr;

mod ffi {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    include!(concat!(env!("OUT_DIR"), "/mnn_c.rs"));
}
pub use ffi::*;
impl DimensionType {
    pub const NHWC: Self = Self::TENSORFLOW;
    pub const NCHW: Self = Self::CAFFE;
    pub const NC4HW4: Self = Self::CAFFE_C4;
}

impl halide_type_t {
    pub fn new(code: halide_type_code_t, bits: u8) -> Self {
        Self::new_with_lanes(code, bits, 1)
    }
    pub fn new_with_lanes(code: halide_type_code_t, bits: u8, lanes: u16) -> Self {
        Self {
            code: code as u8,
            bits,
            lanes,
        }
    }
}

pub fn halide_type_of<T: HalideType>() -> halide_type_t {
    T::halide_type_of()
}

pub trait HalideType: seal::Sealed {
    fn halide_type_of() -> halide_type_t;
}
mod seal {
    pub trait Sealed {}
}

macro_rules! halide_types {
    ($($t:ty => $ht:expr),*) => {
        $(
            impl seal::Sealed for $t {}
            impl HalideType for $t {
                fn halide_type_of() -> halide_type_t {
                    $ht
                }
            }
        )*
    };
}

halide_types! {
    f32 => halide_type_t::new(halide_type_code_t::halide_type_float, 32),
    f64 => halide_type_t::new(halide_type_code_t::halide_type_float, 64),
    bool => halide_type_t::new(halide_type_code_t::halide_type_uint, 1),
    u8 => halide_type_t::new(halide_type_code_t::halide_type_uint, 8),
    u16 => halide_type_t::new(halide_type_code_t::halide_type_uint, 16),
    u32 => halide_type_t::new(halide_type_code_t::halide_type_uint, 32),
    u64 => halide_type_t::new(halide_type_code_t::halide_type_uint, 64),
    i8 => halide_type_t::new(halide_type_code_t::halide_type_int, 8),
    i16 => halide_type_t::new(halide_type_code_t::halide_type_int, 16),
    i32 => halide_type_t::new(halide_type_code_t::halide_type_int, 32),
    i64 => halide_type_t::new(halide_type_code_t::halide_type_int, 64)
}

impl Drop for CString {
    fn drop(&mut self) {
        unsafe { destroyCString(self.as_ptr_mut()) }
    }
}

impl CString {
    pub fn as_ptr(&self) -> *const CString {
        core::ptr::addr_of!(*self)
    }

    pub fn as_ptr_mut(&mut self) -> *mut CString {
        core::ptr::addr_of_mut!(*self)
    }
    pub unsafe fn to_cstr(&self) -> &CStr {
        unsafe { std::ffi::CStr::from_ptr(self.data) }
    }
}

impl AsRef<[i32]> for TensorShape {
    fn as_ref(&self) -> &[i32] {
        &self.shape[..self.size]
    }
}

impl halide_type_code_t {
    pub unsafe fn from_u32(code: u32) -> Self {
        unsafe { std::mem::transmute(code) }
    }
}
