//! Tensor
//!
//! A tensor is a multi-dimensional array used in MNN.
//!
//! There are two types of tensors
//! 1. Host tensor ( CPU )
//! 2. Device tensor ( GPU )
//!
//! You cannot directly read/write the data of a device tensor. You need to copy the data to a host tensor first.
//! For example:
//! ```rust
//! use mnn::*;
//! let mut tensor = Tensor::<Host>::new([1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DimensionType::NHWC);
//! let mut device_tensor = Tensor::<Device>::new([1, 2, 3], &[0.0; 6], DimensionType::NHWC);
//! device_tensor.copy_from_host_tensor(tensor);
//! ```

use crate::prelude::*;
use mnn_sys::*;

/// This is the main Tensor Struct
/// This can hold both Host and Device tensors  
/// With any of the types which implement [HalideType]
#[repr(transparent)]
pub struct Tensor<TT> {
    pub(crate) tensor: TensorRef<'static, TT>,
    pub(crate) __marker: PhantomData<TT>,
}

impl<TT> Drop for Tensor<TT> {
    fn drop(&mut self) {
        unsafe {
            Tensor_destroy(self.tensor.tensor);
        }
    }
}

#[repr(transparent)]
pub struct TensorRef<'a, TT> {
    pub(crate) tensor: *mut mnn_sys::Tensor,
    pub(crate) __marker: PhantomData<(&'a (), TT)>,
}

impl<TT> core::ops::Deref for Tensor<TT> {
    type Target = TensorRef<'static, TT>;

    fn deref(&self) -> &Self::Target {
        &self.tensor
    }
}

impl<TT> core::ops::DerefMut for Tensor<TT> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.tensor
    }
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

impl<'a, TT> TensorRef<'a, TT> {
    pub unsafe fn from_ptr(tensor: *mut mnn_sys::Tensor) -> Self {
        Self {
            tensor,
            __marker: PhantomData,
        }
    }
    pub fn copy_from_host_tensor(&mut self, tensor: &Tensor<Host>) -> Result<()> {
        let ret = unsafe { Tensor_copyFromHostTensor(self.tensor, tensor.tensor.tensor) };
        crate::ensure!(ret != 0, ErrorKind::TensorCopyFailed);
        Ok(())
    }

    pub fn copy_to_host_tensor(&self, tensor: &mut Tensor<Host>) -> Result<()> {
        let ret = unsafe { Tensor_copyToHostTensor(self.tensor, tensor.tensor.tensor) };
        crate::ensure!(ret != 0, ErrorKind::TensorCopyFailed);
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
            tensor: unsafe { TensorRef::from_ptr(tensor) },
            __marker: PhantomData,
        }
    }

    pub fn wait(&self, map_type: MapType, finish: bool) {
        unsafe {
            Tensor_wait(self.tensor, map_type, finish as i32);
        }
    }
}

impl Tensor<Host> {
    pub fn new<T: HalideType>(
        shape: impl TensorShape,
        data: &[T],
        dim_type: DimensionType,
    ) -> Result<Self> {
        let shape = shape.to_shape();
        let shape_size = shape.iter().product::<i32>() as usize;
        ensure!(
            shape_size != data.len(),
            ErrorKind::SizeMismatch {
                expected: shape_size,
                got: data.len()
            }
        );
        let tensor = unsafe {
            Tensor_createWith(
                shape.as_slice().as_ptr().cast(),
                shape.len(),
                halide_type_of::<T>(),
                data.as_ptr().cast::<c_void>().cast_mut(),
                dim_type,
            )
        };
        Ok(Self {
            tensor: unsafe { TensorRef::from_ptr(tensor) },
            __marker: PhantomData,
        })
    }

    pub fn host<T: HalideType>(&self) -> &[T] {
        let size = self.element_size();

        let result = unsafe {
            let data = Tensor_host(self.tensor.tensor).cast();
            core::slice::from_raw_parts(data, size)
        };
        result
    }

    pub fn host_mut<T: HalideType>(&mut self) -> &mut [T] {
        let size = self.element_size();

        let result = unsafe {
            let data: *mut T = Tensor_host_mut(self.tensor.tensor).cast();
            debug_assert!(!data.is_null());
            core::slice::from_raw_parts_mut(data, size)
        };
        result
    }
}

impl Tensor<Device> {
    /// Create a new device tensor
    ///
    /// Note: The data is not copied to the device tensor directly.
    /// You need to call `copy_from_host_tensor` to copy the data to the device tensor by creating
    /// a host tensor
    pub fn new<T: HalideType>(shape: impl TensorShape, dim_type: DimensionType) -> Result<Self> {
        // debug_assert_eq!(shape.iter().product::<i32>() as usize, size);
        let shape = shape.to_shape();
        let tensor = unsafe {
            Tensor_createDevice(
                shape.as_slice().as_ptr().cast(),
                shape.len(),
                halide_type_of::<T>(),
                dim_type,
            )
        };

        Ok(Self {
            tensor: unsafe { TensorRef::from_ptr(tensor) },
            __marker: PhantomData,
        })
    }
}
