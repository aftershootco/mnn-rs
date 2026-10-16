use super::{Tensor, *};

/// This is modeled as a reference to a tensor, you should never be able to dereference this.
/// At any point you should only have &TensorRef or &mut TensorRef, never an owned TensorRef.
#[repr(transparent)]
pub struct TensorRef<H, M>
where
    H: HalideType,
    M: TensorMachine,
{
    __marker: PhantomData<(M, mnn_sys::Tensor, [H])>,
}

impl<H, M> core::fmt::Debug for TensorRef<H, M>
where
    H: HalideType,
    M: TensorMachine,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let dynamic = self.is_dynamic_unsized();
        // The element count and byte size are meaningless (and computed from negative
        // extents) until a dynamic tensor has been resized.
        let elements = (!dynamic).then(|| self.element_size());
        let size = (!dynamic).then(|| self.size());
        f.debug_struct("TensorRef")
            .field("shape", &self.shape())
            .field("device_id", &self.device_id())
            .field("dimensions", &self.get_dimension_type())
            .field("elements", &elements)
            .field("dynamic", &dynamic)
            .field("size", &size)
            .finish()
    }
}

impl<H: HalideType, M: TensorMachine> TensorRef<H, M> {
    /// Get a raw pointer to the underlying MNN tensor
    pub(crate) fn as_ptr(&self) -> *mut mnn_sys::Tensor {
        self as *const Self as *mut mnn_sys::Tensor
    }

    /// Copies the data from a host tensor to the self.as_ptr()
    ///
    /// The tensors must have the same logical shape; MNN converts between layouts
    /// (e.g. NCHW <-> NC4HW4) as part of the copy.
    pub fn copy_from_host_tensor(&mut self, tensor: &TensorRef<H, Host>) -> Result<()> {
        AnyTensorRef::copy_from_host_tensor(self, tensor)
    }

    /// Copies the data from the self.as_ptr() to a host tensor
    ///
    /// The tensors must have the same logical shape; MNN converts between layouts
    /// (e.g. NCHW <-> NC4HW4) as part of the copy.
    pub fn copy_to_host_tensor(&self, tensor: &mut TensorRef<H, Host>) -> Result<()> {
        AnyTensorRef::copy_to_host_tensor(self, tensor)
    }
}

impl<H, M> core::ops::Deref for TensorRef<H, M>
where
    H: HalideType,
    M: TensorMachine,
{
    type Target = AnyTensorRef;
    fn deref(&self) -> &Self::Target {
        unsafe { &*(self.as_ptr() as *const AnyTensorRef) }
    }
}

impl<H, M> core::ops::DerefMut for TensorRef<H, M>
where
    H: HalideType,
    M: TensorMachine,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *(self.as_ptr().cast()) }
    }
}

impl<T> TensorRef<T, Device>
where
    T: HalideType,
{
    /// Try to wait for the device tensor to finish processing
    pub fn wait(&self, map_type: MapType, finish: bool) {
        unsafe {
            mnn_sys::Tensor_wait(self.as_ptr(), map_type, finish as i32);
        }
    }

    /// Create a host tensor from the device tensor with same dimensions and data type and
    /// optionally copy the data from the device tensor
    ///
    /// The host tensor uses the unpacked layout (NCHW / NHWC) even when the device tensor is
    /// stored as NC4HW4, so `host()` yields the elements in logical order.
    pub fn create_host_tensor_from_device(&self, copy_data: bool) -> super::Tensor<Owned<T>, Host> {
        let shape = self.shape();
        assert!(
            !self.is_dynamic_unsized(),
            "Cannot create a host tensor from a device tensor with a dynamic shape {shape:?}; resize the session first"
        );
        let dm_type = self.get_dimension_type();
        let mut out = super::Tensor::new(shape, dm_type);

        if copy_data {
            self.copy_to_host_tensor(&mut out)
                .expect("Failed to copy data from device tensor");
        }
        out
    }
}

/// Construct a tensor reference from a raw pointer to an MNN tensor
///
/// # Safety
/// The caller must ensure that the provided pointer is valid and points to a properly initialized MNN
#[inline]
pub unsafe fn from_raw_parts<'a, H, M>(tensor: *const mnn_sys::Tensor) -> &'a TensorRef<H, M>
where
    H: HalideType,
    M: TensorMachine,
{
    unsafe { &*tensor.cast::<TensorRef<H, M>>() }
}

/// Construct a mutable tensor reference from a raw pointer to an MNN tensor
///
/// # Safety
/// The caller must ensure that the provided pointer is valid and points to a properly initialized
#[inline]
pub unsafe fn from_raw_parts_mut<'a, H, M>(tensor: *mut mnn_sys::Tensor) -> &'a mut TensorRef<H, M>
where
    H: HalideType,
    M: TensorMachine,
{
    unsafe { &mut *tensor.cast::<TensorRef<H, M>>() }
}

impl<S, M, H> core::ops::Deref for Tensor<S, M, H>
where
    S: TensorType<H = H>,
    H: HalideType,
    M: TensorMachine,
{
    type Target = TensorRef<H, M>;

    fn deref(&self) -> &Self::Target {
        unsafe { from_raw_parts(self.tensor) }
    }
}

/// Only tensors that own their data or hold a mutable borrow of it can be mutated through
/// the [`TensorRef`]; a `Tensor<View<&H>, _>` built over a shared slice must not be able to
/// write through that slice.
impl<S, M, H> core::ops::DerefMut for Tensor<S, M, H>
where
    S: MutableTensorType<H = H>,
    H: HalideType,
    M: TensorMachine,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { from_raw_parts_mut(self.tensor) }
    }
}

impl<T, M> TensorRef<T, M>
where
    T: HalideType,
    M: TensorMachine,
{
    /// Fill the tensor with the specified value
    ///
    /// Fails if the tensor has a different element type, has not been resized yet (dynamic
    /// shape) or, for device tensors, if MNN refuses the copy (e.g. no backend is attached).
    pub fn fill(&mut self, value: T) -> Result<()>
    where
        T: Copy,
    {
        ensure!(
            self.is_type_of::<T>(),
            ErrorKind::HalideTypeMismatch {
                got: std::any::type_name::<T>(),
            }
        );
        ensure!(!self.is_dynamic_unsized(), ErrorKind::DynamicTensorError);
        if M::host() {
            let size = self.element_size();
            let result: &mut [T] = unsafe {
                let data = mnn_sys::Tensor_host_mut(self.as_ptr()).cast();
                core::slice::from_raw_parts_mut(data, size)
            };
            result.fill(value);
        } else {
            let shape = self.shape();
            let dm_type = self.get_dimension_type();
            let mut host: TensorOwned<T, Host> = Tensor::new(shape, dm_type);
            host.fill(value)?;
            self.copy_from_host_tensor(&host)?;
        }
        Ok(())
    }
}

impl<T> TensorRef<T, Host>
where
    T: HalideType,
{
    /// Try to map the device tensor to the host memory and get the slice
    pub fn try_host(&self) -> Result<&[T]> {
        let size = self.element_size();
        ensure!(
            self.is_type_of::<T>(),
            ErrorKind::HalideTypeMismatch {
                got: std::any::type_name::<T>(),
            }
        );
        let result = unsafe {
            let data = mnn_sys::Tensor_host(self.as_ptr()).cast();
            core::slice::from_raw_parts(data, size)
        };
        Ok(result)
    }

    /// Try to map the device tensor to the host memory and get the mutable slice
    pub fn try_host_mut(&mut self) -> Result<&mut [T]> {
        let size = self.element_size();
        ensure!(
            self.is_type_of::<T>(),
            ErrorKind::HalideTypeMismatch {
                got: std::any::type_name::<T>(),
            }
        );

        let result = unsafe {
            let data: *mut T = mnn_sys::Tensor_host_mut(self.as_ptr()).cast();
            debug_assert!(!data.is_null());
            core::slice::from_raw_parts_mut(data, size)
        };
        Ok(result)
    }

    /// Get the host memory slice of the tensor
    pub fn host(&self) -> &[T] {
        self.try_host().expect("Failed to get tensor host")
    }

    /// Get the mutable host memory slice of the tensor
    pub fn host_mut(&mut self) -> &mut [T] {
        self.try_host_mut().expect("Failed to get tensor host_mut")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_from_host_tensor_reports_failure() {
        // A plain host tensor has no backend, so MNN refuses the copy; the error must not be
        // swallowed.
        let src = TensorOwned::<f32, Host>::from_shape_data(
            [1, 2, 3],
            DimensionType::Caffe,
            &[1., 2., 3., 4., 5., 6.],
        );
        let mut dst = TensorOwned::<f32, Host>::new([1, 2, 3], DimensionType::Caffe);
        dst.copy_from_host_tensor(&src)
            .expect_err("copy without a backend must fail");
        src.copy_to_host_tensor(&mut dst)
            .expect_err("copy without a backend must fail");
    }

    #[test]
    fn fill_on_device_tensor_without_backend_reports_failure() {
        let mut dev = TensorOwned::<f32, Device>::new([1, 2, 3], DimensionType::Caffe);
        dev.fill(1.0).expect_err("fill without a backend must fail");
    }

    #[test]
    fn fill_rejects_wrong_element_type() {
        let t = TensorOwned::<f32, Host>::new([1, 2, 3], DimensionType::Caffe);
        // Reinterpret the typed reference to request a fill with the wrong element type.
        let wrong: &mut TensorRef<i32, Host> = unsafe { from_raw_parts_mut(t.as_ptr()) };
        wrong
            .fill(1)
            .expect_err("filling an f32 tensor with i32 must fail");
    }

    #[test]
    fn debug_of_dynamic_tensor_does_not_panic() {
        // A device tensor is never allocated, so a dynamic (-1) extent is representable.
        let t = TensorOwned::<f32, Device>::new([1, -1, 3], DimensionType::Caffe);
        let s = format!("{:?}", &*t);
        assert!(s.contains("dynamic: true"), "{s}");
        assert!(s.contains("elements: None"), "{s}");
    }
}
