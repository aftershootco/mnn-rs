use crate::prelude::*;
use core::marker::PhantomData;
use mnn_sys::HalideType;
/// A raw tensor type that doesn't have any guarantees
/// and will be unconditionally dropped
#[repr(transparent)]
pub struct RawTensor<'r> {
    pub(crate) inner: *mut mnn_sys::Tensor,
    pub(crate) __marker: PhantomData<&'r ()>,
}

// impl<'r> core::ops::Drop for RawTensor<'r> {
//     fn drop(&mut self) {
//         unsafe {
//             mnn_sys::Tensor_destroy(self.inner);
//         }
//     }
// }

impl RawTensor<'_> {
    /// Creates a new host tensor from the device tensor
    pub fn create_host_tensor_from_device(&self, copy_data: bool) -> RawTensor<'static> {
        let tensor =
            unsafe { mnn_sys::Tensor_createHostTensorFromDevice(self.inner, copy_data as i32) };
        // crate::ensure!(!tensor.is_null(), ErrorKind::TensorError);
        assert!(!tensor.is_null());
        RawTensor {
            inner: tensor,
            __marker: PhantomData,
        }
    }

    /// Copies the data from a host tensor to the self tensor
    pub fn copy_from_host_tensor(&mut self, tensor: &RawTensor) -> Result<()> {
        let ret = unsafe { mnn_sys::Tensor_copyFromHostTensor(self.inner, tensor.inner) };
        crate::ensure!(ret != 0, ErrorKind::TensorCopyFailed(ret));
        Ok(())
    }

    /// Copies the data from the self tensor to a host tensor
    pub fn copy_to_host_tensor(&self, tensor: &mut RawTensor) -> Result<()> {
        let ret = unsafe { mnn_sys::Tensor_copyToHostTensor(self.inner, tensor.inner) };
        crate::ensure!(ret != 0, ErrorKind::TensorCopyFailed(ret));
        Ok(())
    }

    /// Returns the shape of the tensor
    pub fn shape(&self) -> crate::TensorShape {
        unsafe { mnn_sys::Tensor_shape(self.inner) }.into()
    }

    /// Returns the dimension type of the tensor
    pub fn get_dimension_type(&self) -> super::DimensionType {
        debug_assert!(!self.inner.is_null());
        From::from(unsafe { mnn_sys::Tensor_getDimensionType(self.inner) })
    }

    /// Cleans up the tensor by calling the destructor of the tensor
    pub fn destroy(self) {
        unsafe {
            mnn_sys::Tensor_destroy(self.inner);
        }
    }

    /// Returns the size of the tensor when counted by bytes
    pub fn size(&self) -> usize {
        unsafe { mnn_sys::Tensor_usize(self.inner) }
    }

    /// Returns the size of the tensor when counted by elements
    pub fn element_size(&self) -> usize {
        unsafe { mnn_sys::Tensor_elementSize(self.inner) as usize }
    }

    /// Returns the number of dimensions of the tensor
    pub fn dimensions(&self) -> usize {
        unsafe { mnn_sys::Tensor_dimensions(self.inner) as usize }
    }

    /// Returns the width of the tensor
    pub fn width(&self) -> u32 {
        unsafe { mnn_sys::Tensor_width(self.inner) as u32 }
    }

    /// Returns the height of the tensor
    pub fn height(&self) -> u32 {
        unsafe { mnn_sys::Tensor_height(self.inner) as u32 }
    }

    /// Returns the channel of the tensor
    pub fn channel(&self) -> u32 {
        unsafe { mnn_sys::Tensor_channel(self.inner) as u32 }
    }

    /// Returns true if the tensor is unsized and dynamic (needs to be resized to work)
    pub fn is_dynamic_unsized(&self) -> bool {
        self.shape().as_ref().contains(&-1)
    }

    /// Waits for the tensor to be ready
    pub fn wait(&self, map_type: MapType, finish: bool) {
        unsafe {
            mnn_sys::Tensor_wait(self.inner, map_type, finish as i32);
        }
    }

    /// # Safety
    /// This is very unsafe do not use this unless you know what you are doing
    /// Gives a raw pointer to the tensor's data
    /// P.S. I don't know what I'm doing
    pub unsafe fn unchecked_host_ptr(&self) -> *mut c_void {
        debug_assert!(!self.inner.is_null());
        let data = unsafe { mnn_sys::Tensor_host_mut(self.inner) };
        debug_assert!(!data.is_null());
        data
    }

    /// # Safety
    /// This is very unsafe do not use this unless you know what you are doing
    /// Gives a mutable byte slice to the tensor's data
    pub unsafe fn unchecked_host_bytes(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.unchecked_host_ptr().cast(), self.size()) }
    }

    /// # Safety
    /// This is very unsafe do not use this unless you know what you are doing
    pub unsafe fn to_concrete<T: super::TensorType, M: super::TensorMachine>(
        self,
    ) -> super::Tensor<T, M>
    where
        T::H: HalideType,
    {
        unsafe { super::Tensor::from_ptr(self.inner) }
    }

    pub(crate) fn from_ptr(inner: *mut mnn_sys::Tensor) -> Self {
        Self {
            inner,
            __marker: PhantomData,
        }
    }
}
