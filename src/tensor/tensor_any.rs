use super::*;
/// This is a zero sized type
#[repr(transparent)]
pub struct AnyTensorRef {
    __marker: PhantomData<[u8]>,
}

impl AnyTensorRef {
    /// Get a raw pointer to the underlying MNN tensor
    pub(crate) fn as_ptr(&self) -> *mut mnn_sys::Tensor {
        self as *const Self as *mut mnn_sys::Tensor
    }

    pub(crate) unsafe fn from_ptr<'s>(tensor: *mut mnn_sys::Tensor) -> &'s Self {
        unsafe { &*tensor.cast() }
    }

    pub(crate) unsafe fn from_mut_ptr<'s>(tensor: *mut mnn_sys::Tensor) -> &'s mut Self {
        unsafe { &mut *tensor.cast() }
    }

    /// Get the device id of the tensor
    pub fn device_id(&self) -> u64 {
        unsafe { mnn_sys::Tensor_deviceId(self.as_ptr()) }
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> TensorShape {
        unsafe { mnn_sys::Tensor_shape(self.as_ptr()) }.into()
    }

    /// Get the dimensions of the tensor
    #[doc(alias = "dims")]
    pub fn dimensions(&self) -> usize {
        unsafe { mnn_sys::Tensor_dimensions(self.as_ptr()) as usize }
    }

    /// Get the width of the tensor
    pub fn width(&self) -> u32 {
        unsafe { mnn_sys::Tensor_width(self.as_ptr()) as u32 }
    }

    /// Get the height of the tensor
    pub fn height(&self) -> u32 {
        unsafe { mnn_sys::Tensor_height(self.as_ptr()) as u32 }
    }

    /// Get the channel size of the tensor
    pub fn channel(&self) -> u32 {
        unsafe { mnn_sys::Tensor_channel(self.as_ptr()) as u32 }
    }

    /// Get the batch size of the tensor
    pub fn batch(&self) -> u32 {
        unsafe { mnn_sys::Tensor_batch(self.as_ptr()) as u32 }
    }

    /// Get the size of the tensor when counted by bytes
    ///
    /// Note that for packed layouts (NC4HW4) this includes the padding MNN adds to the
    /// channel dimension, so it can be larger than the product of the shape.
    pub fn size(&self) -> usize {
        unsafe { mnn_sys::Tensor_usize(self.as_ptr()) }
    }

    /// Get the size of the tensor when counted by elements
    ///
    /// Note that for packed layouts (NC4HW4) this includes the padding MNN adds to the
    /// channel dimension, so it can be larger than the product of the shape.
    pub fn element_size(&self) -> usize {
        unsafe { mnn_sys::Tensor_elementSize(self.as_ptr()) as usize }
    }

    /// Get the halide data type of the tensor
    pub fn get_type(&self) -> mnn_sys::halide_type_t {
        unsafe { mnn_sys::Tensor_getType(self.as_ptr()) }
    }

    /// Check if the tensor is of the specified data type
    pub fn is_type_of<Ha: HalideType>(&self) -> bool {
        let htc = mnn_sys::halide_type_of::<Ha>();
        unsafe { mnn_sys::Tensor_isTypeOf(self.as_ptr(), htc) }
    }

    /// Check if both tensors have the same halide data type
    pub fn is_same_type(&self, other: &AnyTensorRef) -> bool {
        unsafe { mnn_sys::Tensor_isTypeOf(self.as_ptr(), other.get_type()) }
    }

    /// Get the dimension type of the tensor
    pub fn get_dimension_type(&self) -> DimensionType {
        debug_assert!(!self.as_ptr().is_null());
        From::from(unsafe { mnn_sys::Tensor_getDimensionType(self.as_ptr()) })
    }

    /// Check if the tensor is dynamic and needs resizing
    pub fn is_dynamic_unsized(&self) -> bool {
        self.shape().as_ref().contains(&-1)
    }

    /// Check that `other` has the same element type and logical shape as `self`.
    ///
    /// Byte sizes are deliberately not compared: a packed layout (e.g. NC4HW4) is
    /// allowed to differ in allocation size from the host layout it converts into.
    fn ensure_compatible(&self, other: &AnyTensorRef) -> Result<()> {
        crate::ensure!(
            self.is_same_type(other),
            ErrorKind::TensorError;
            format!(
                "Tensor element types do not match: {:?} vs {:?}",
                self.get_type(),
                other.get_type()
            )
        );
        crate::ensure!(
            self.shape().as_ref() == other.shape().as_ref(),
            ErrorKind::SizeMismatch {
                expected: self.element_size(),
                got: other.element_size()
            };
            format!(
                "Tensor shapes do not match: {:?} vs {:?}",
                self.shape(),
                other.shape()
            )
        );
        Ok(())
    }

    /// Copies the data from a host tensor to the self.as_ptr()
    pub fn copy_from_host_tensor(&mut self, tensor: &AnyTensorRef) -> Result<()> {
        self.ensure_compatible(tensor)?;
        let ret = unsafe { mnn_sys::Tensor_copyFromHostTensor(self.as_ptr(), tensor.as_ptr()) };
        crate::ensure!(ret != 0, ErrorKind::TensorCopyFailed(ret));
        Ok(())
    }

    /// Copies the data from the self.as_ptr() to a host tensor
    pub fn copy_to_host_tensor(&self, tensor: &mut AnyTensorRef) -> Result<()> {
        self.ensure_compatible(tensor)?;
        let ret = unsafe { mnn_sys::Tensor_copyToHostTensor(self.as_ptr(), tensor.as_ptr()) };
        crate::ensure!(ret != 0, ErrorKind::TensorCopyFailed(ret));
        Ok(())
    }

    /// Create a host tensor from the device tensor with same dimensions and data type and
    /// optionally copy the data from the device tensor
    ///
    /// # Panics
    /// Panics if the tensor's element type is not `T` or the copy fails.
    pub fn create_host_tensor_from_device<T: HalideType + Copy>(
        &self,
        copy_data: bool,
    ) -> Tensor<Owned<T>, Host> {
        assert!(
            self.is_type_of::<T>(),
            "Tensor element type does not match {}",
            core::any::type_name::<T>()
        );
        let shape = self.shape();
        assert!(
            !self.is_dynamic_unsized(),
            "Cannot create a host tensor from a device tensor with a dynamic shape {shape:?}; resize the session first"
        );
        let dm_type = self.get_dimension_type();
        let mut out = Tensor::new(shape, dm_type);

        if copy_data {
            self.copy_to_host_tensor(unsafe { out.as_any_tensor_mut() })
                .expect("Failed to copy data from device tensor");
        }
        out
    }

    /// Try to wait for the device tensor to finish processing
    pub fn wait(this: &Self, map_type: MapType, finish: bool) {
        unsafe {
            mnn_sys::Tensor_wait(this.as_ptr(), map_type, finish as i32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_rejects_element_type_mismatch() {
        let src = TensorOwned::<f32, Host>::from_shape_data(
            [1, 2, 3],
            DimensionType::Caffe,
            &[1., 2., 3., 4., 5., 6.],
        );
        let mut dst = TensorOwned::<u8, Host>::new([1, 2, 3], DimensionType::Caffe);
        let err = unsafe {
            src.as_any_tensor()
                .copy_to_host_tensor(dst.as_any_tensor_mut())
        }
        .expect_err("copying f32 into u8 must fail");
        assert!(format!("{err}").contains("element types do not match"));
        let err = unsafe {
            dst.as_any_tensor_mut()
                .copy_from_host_tensor(src.as_any_tensor())
        }
        .expect_err("copying f32 into u8 must fail");
        assert!(format!("{err}").contains("element types do not match"));
    }

    #[test]
    fn copy_rejects_shape_mismatch() {
        let src = TensorOwned::<f32, Host>::new([1, 2, 3], DimensionType::Caffe);
        let mut dst = TensorOwned::<f32, Host>::new([1, 3, 2], DimensionType::Caffe);
        let err = unsafe {
            src.as_any_tensor()
                .copy_to_host_tensor(dst.as_any_tensor_mut())
        }
        .expect_err("mismatched shapes must fail");
        assert!(format!("{err}").contains("shapes do not match"));
    }
}
