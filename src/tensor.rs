//! Tensor
//!
//! A tensor is a multi-dimensional array used in MNN.
//!
//! There are two types of tensors
//! 1. Host tensor ( CPU )
//! 2. Device tensor ( GPU )
//!
//! You cannot directly access the data of a device tensor. You need to copy the data to a host tensor first.
use anyhow::Result;
use core::marker::PhantomData;
use libc::c_void;
use mnn_sys::*;

pub struct Tensor<TT> {
    pub(crate) tensor: *mut mnn_sys::Tensor,
    pub(crate) __marker: PhantomData<TT>,
}

pub trait TensorType {
    fn tensor_type() -> Self;
}
pub struct Host;
pub struct Device;
impl TensorType for Host {
    fn tensor_type() -> Self {
        Self
    }
}
impl TensorType for Device {
    fn tensor_type() -> Self {
        Self
    }
}

pub trait TensorShape {
    fn to_shape(&self) -> Vec<i32>;
}
macro_rules! tensor_shape {
    ($($name:ty),*) => {
        $(
            impl TensorShape for $name {
                fn to_shape(&self) -> Vec<i32> {
                    self.to_vec()
                }
            }
        )*
    }
}
tensor_shape!([i32; 1], [i32; 2], [i32; 3], [i32; 4], Vec<i32>);

impl<TT> Tensor<TT> {
    pub fn copy_from_host_tensor(&mut self, tensor: &Tensor<Host>) -> Result<()> {
        let ret = unsafe { Tensor_copyFromHostTensor(self.tensor, tensor.tensor) };
        let ret = ret != 0;
        if !ret {
            anyhow::bail!("Tensor_copyFromHostTensor failed");
        }
        // if ret != ErrorCode::ERROR_CODE_NO_ERROR as i32 {
        //     anyhow::bail!("Tensor_copyFromHostTensor failed {ret:?}");
        // }
        Ok(())
    }

    pub fn copy_to_host_tensor(&self, tensor: &mut Tensor<Host>) -> Result<()> {
        let ret = unsafe { Tensor_copyToHostTensor(self.tensor, tensor.tensor) };
        let ret = ret != 0;
        if !ret {
            anyhow::bail!("Tensor_copyToHostTensor failed");
        }
        Ok(())
    }

    pub fn device_id(&self) -> u64 {
        unsafe { Tensor_deviceId(self.tensor) }
    }

    pub fn shape(&self) -> mnn_sys::TensorShape {
        unsafe { Tensor_shape(self.tensor) }
    }

    pub fn dimensions(&self) -> usize {
        unsafe { Tensor_dimensions(self.tensor) as usize }
    }

    pub fn size(&self) -> usize {
        unsafe { Tensor_size(self.tensor) as usize }
    }

    pub fn element_size(&self) -> usize {
        unsafe { Tensor_elementSize(self.tensor) as usize }
    }

    pub fn print_shape(&self) {
        unsafe {
            Tensor_printShape(self.tensor);
        }
    }

    pub unsafe fn halide_buffer<T: HalideType>(&self) -> *const halide_buffer_t {
        Tensor_buffer(self.tensor)
    }

    pub unsafe fn halide_buffer_mut<T: HalideType>(&self) -> *mut halide_buffer_t {
        Tensor_buffer_mut(self.tensor)
    }

    pub fn get_diemension_type(&self) -> mnn_sys::DimensionType {
        unsafe { Tensor_getDimensionType(self.tensor) }
    }

    pub fn get_type(&self) -> mnn_sys::halide_type_c {
        let type_ = unsafe { Tensor_getType(self.tensor) };
        type_
    }

    pub fn create_host_tensor_from_device(&self, copy_data: bool) -> Tensor<Host> {
        let tensor = unsafe { Tensor_createHostTensorFromDevice(self.tensor, copy_data as i32) };
        debug_assert!(!tensor.is_null());
        Tensor {
            tensor,
            __marker: PhantomData,
        }
    }
}

impl Tensor<Host> {
    pub fn new<T: HalideType>(
        shape: impl TensorShape,
        data: &[T],
        dim_type: DimensionType,
    ) -> Self {
        assert_eq!(
            shape.to_shape().iter().product::<i32>() as usize,
            data.len()
        );
        let shape = shape.to_shape();
        let tensor = unsafe {
            Tensor_createWith(
                shape.as_slice().as_ptr().cast(),
                shape.len(),
                halide_type_of::<T>(),
                data.as_ptr().cast::<c_void>().cast_mut(),
                dim_type,
            )
        };
        Self {
            tensor,
            __marker: PhantomData,
        }
    }

    pub fn host<T: HalideType>(&self) -> &[T] {
        let size = self.element_size();

        let result = unsafe {
            let data = Tensor_host(self.tensor).cast();
            core::slice::from_raw_parts(data, size)
        };
        result
    }

    pub fn host_mut<T: HalideType>(&mut self) -> &mut [T] {
        let size = self.element_size();

        let result = unsafe {
            let data: *mut T = Tensor_host_mut(self.tensor).cast();
            debug_assert!(!data.is_null());
            core::slice::from_raw_parts_mut(data, size)
        };
        result
    }
}

impl<TT> Drop for Tensor<TT> {
    fn drop(&mut self) {
        unsafe {
            Tensor_destroy(self.tensor);
        }
    }
}
