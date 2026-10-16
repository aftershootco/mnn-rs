use crate::prelude::*;
use core::marker::PhantomData;
pub mod list;
mod tensor_any;
mod tensor_ref;
pub use tensor_any::AnyTensorRef;
pub use tensor_ref::{TensorRef, from_raw_parts, from_raw_parts_mut};

use mnn_sys::HalideType;

/// A tensor that can be owned by the host or device
pub type TensorOwned<H, M> = Tensor<Owned<H>, M>;
/// A tensor that can be borrowed from the host or device
pub type TensorView<'t, H, M> = Tensor<View<&'t H>, M>;
/// A tensor that can be borrowed mutably from the host or device
pub type TensorViewMut<'t, H, M> = Tensor<View<&'t mut H>, M>;

mod seal {
    pub trait Sealed {}
}
impl seal::Sealed for Host {}
impl seal::Sealed for Device {}
impl<T> seal::Sealed for View<T> {}
impl<T> seal::Sealed for Owned<T> {}

/// A trait to represent the type of a tensor
pub trait TensorType: seal::Sealed {
    /// The type of the tensor data
    type H;
    /// Check if the tensor is owned
    fn owned() -> bool;
    /// Check if the tensor is borrowed
    fn borrowed() -> bool {
        !Self::owned()
    }
}

/// A trait to represent a mutable tensor type
pub trait MutableTensorType: TensorType + seal::Sealed {}

impl<H> MutableTensorType for Owned<H> {}
impl<H> MutableTensorType for View<&mut H> {}

impl<H> TensorType for Owned<H> {
    type H = H;
    fn owned() -> bool {
        true
    }
}

impl<H> TensorType for View<&H> {
    type H = H;
    fn owned() -> bool {
        false
    }
}

impl<H> TensorType for View<&mut H> {
    type H = H;
    fn owned() -> bool {
        false
    }
}

/// A trait to represent the device type of a tensor
pub trait TensorMachine: seal::Sealed {
    /// Check if the tensor is owned by the device
    fn device() -> bool;
    /// Check if the tensor is owned by the host
    fn host() -> bool {
        !Self::device()
    }
}

impl TensorMachine for Host {
    fn device() -> bool {
        false
    }
}

impl TensorMachine for Device {
    fn device() -> bool {
        true
    }
}

/// A tensor that is owned by the cpu / host platform
#[non_exhaustive]
#[derive(Debug)]
pub struct Host {}
/// A tensor that is owned by the device / gpu platform
#[non_exhaustive]
#[derive(Debug)]
pub struct Device {}

/// A reference to a any tensor
#[repr(transparent)]
#[derive(Debug)]
pub struct View<T> {
    pub(crate) __marker: PhantomData<[T]>,
}
/// A tensor that is owned by the host / device platform
#[repr(transparent)]
#[derive(Debug)]
pub struct Owned<T> {
    pub(crate) __marker: PhantomData<T>,
}

/// A generic tensor that can of host / device / owned / borrowed
#[derive(Debug)]
pub struct Tensor<S, M, A = <S as TensorType>::H>
where
    A: HalideType,
    M: TensorMachine,
    S: TensorType<H = A>,
{
    pub(crate) tensor: *mut mnn_sys::Tensor,
    /// Whether this `Tensor` owns the C++ `MNN::Tensor` object behind `tensor`.
    ///
    /// Owned tensors and the views created by [`Tensor::borrowed`] / [`Tensor::borrowed_mut`]
    /// allocate their own `MNN::Tensor` (the latter over user provided memory, which MNN never
    /// frees) and must destroy it. Views created from another `Tensor` through
    /// [`Tensor::view`] / [`Tensor::view_mut`] share the C++ object with their parent and must
    /// not.
    destroy_on_drop: bool,
    __marker: PhantomData<(S, M, A)>,
}

impl<S, M, A> Drop for Tensor<S, M, A>
where
    A: HalideType,
    S: TensorType<H = A>,
    M: TensorMachine,
{
    fn drop(&mut self) {
        if self.destroy_on_drop {
            unsafe {
                mnn_sys::Tensor_destroy(self.tensor);
            }
        }
    }
}

impl<'a, T: TensorType<H = H>, H: HalideType, M: TensorMachine> Tensor<T, M> {
    /// Get's a reference to an owned host tensor
    pub fn view(&'a self) -> Tensor<View<&'a H>, M> {
        Tensor {
            tensor: self.tensor,
            destroy_on_drop: false,
            __marker: PhantomData,
        }
    }
}

impl<'a, H: HalideType, M: TensorMachine> Tensor<View<&'a H>, M> {
    /// Reborrows the tensor to get rid of self's lifetime the tensor while using the lifetime inside of the TensorView
    pub fn reborrow(&self) -> Tensor<View<&'a H>, M> {
        Tensor {
            tensor: self.tensor,
            destroy_on_drop: false,
            __marker: PhantomData,
        }
    }
}

impl<T: MutableTensorType<H = H>, H: HalideType, M: TensorMachine> Tensor<T, M> {
    /// Get's a mutable reference to an owned host tensor
    pub fn view_mut(&mut self) -> Tensor<View<&mut H>, M> {
        Tensor {
            tensor: self.tensor,
            destroy_on_drop: false,
            __marker: PhantomData,
        }
    }
}

impl<H: HalideType, M: TensorMachine> Tensor<View<&mut H>, M> {
    /// Reborrows the tensor as mutable.
    ///
    /// The returned view borrows `self` mutably for as long as it lives, exactly like
    /// reborrowing a `&mut T`, so the original and the reborrow can never be used at the same
    /// time.
    pub fn reborrow_mut(&mut self) -> Tensor<View<&mut H>, M> {
        Tensor {
            tensor: self.tensor,
            destroy_on_drop: false,
            __marker: PhantomData,
        }
    }
}

/// The type of the tensor dimension
/// If you are manually specifying the shapes then this doesn't really matter
/// N -> Batch size
/// C -> Channel
/// H -> Height
/// W -> Width
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DimensionType {
    /// Caffe style dimensions or NCHW
    #[default]
    Caffe,
    /// Caffe style dimensions with channel packed in 4 bytes or NC4HW4
    CaffeC4,
    /// Tensorflow style dimensions or NHWC
    TensorFlow,
}

impl DimensionType {
    /// Tensorflow style dimensions or NHWC
    pub const NHWC: Self = Self::TensorFlow;
    /// Caffe style dimensions or NCHW
    pub const NCHW: Self = Self::Caffe;
    /// Caffe style dimensions with channel packed in 4 bytes or NC4HW4
    pub const NC4HW4: Self = Self::CaffeC4;
    pub(crate) fn to_mnn_sys(self) -> mnn_sys::DimensionType {
        match self {
            DimensionType::Caffe => mnn_sys::DimensionType::CAFFE,
            DimensionType::CaffeC4 => mnn_sys::DimensionType::CAFFE_C4,
            DimensionType::TensorFlow => mnn_sys::DimensionType::TENSORFLOW,
        }
    }
}

impl From<mnn_sys::DimensionType> for DimensionType {
    fn from(dm: mnn_sys::DimensionType) -> Self {
        match dm {
            mnn_sys::DimensionType::CAFFE => DimensionType::Caffe,
            mnn_sys::DimensionType::CAFFE_C4 => DimensionType::CaffeC4,
            mnn_sys::DimensionType::TENSORFLOW => DimensionType::TensorFlow,
        }
    }
}

impl<H, T, M> Tensor<T, M>
where
    T: TensorType<H = H>,
    H: HalideType,
    M: TensorMachine,
{
    /// This function constructs a Tensor type from a raw pointer
    ///
    /// The returned tensor destroys the underlying `MNN::Tensor` on drop if it is an owned
    /// tensor type and leaves it alone if it is a view.
    ///# Safety
    /// Since this constructs a Tensor from a raw pointer we have no way to guarantee that it's a
    /// valid tensor or it's lifetime
    pub unsafe fn from_ptr(tensor: *mut mnn_sys::Tensor) -> Self {
        assert!(!tensor.is_null());
        Self {
            tensor,
            destroy_on_drop: T::owned(),
            __marker: PhantomData,
        }
    }

    /// Print the tensor
    pub fn print(&self) {
        unsafe {
            mnn_sys::Tensor_print(self.tensor);
        }
    }

    /// DO not use this function directly
    /// # Safety
    /// This is just provided as a 1:1 compat mostly for possible later use
    pub unsafe fn halide_buffer(&self) -> *const mnn_sys::halide_buffer_t {
        unsafe { mnn_sys::Tensor_buffer(self.tensor) }
    }

    /// Do not use this function directly
    /// # Safety
    /// This is just provided as a 1:1 compat mostly for possible later use
    pub unsafe fn halide_buffer_mut(&self) -> *mut mnn_sys::halide_buffer_t {
        unsafe { mnn_sys::Tensor_buffer_mut(self.tensor) }
    }

    /// Get the data type of the tensor
    pub fn get_type(&self) -> mnn_sys::halide_type_t {
        unsafe { mnn_sys::Tensor_getType(self.tensor) }
    }

    /// # Safety
    ///
    /// Type erase the tensor
    pub unsafe fn as_any_tensor(&self) -> &AnyTensorRef {
        unsafe { AnyTensorRef::from_ptr(self.tensor) }
    }

    /// # Safety
    ///
    /// Type erase the tensor mutably
    pub unsafe fn as_any_tensor_mut(&mut self) -> &mut AnyTensorRef {
        unsafe { AnyTensorRef::from_mut_ptr(self.tensor) }
    }
}

impl<H> Tensor<Owned<H>, Host>
where
    H: HalideType + Copy,
{
    /// Create a new tensor with the specified shape and dimension type and fill it with the data from the provided vector
    ///
    /// # Panics
    /// Panics if the shape is dynamic or the number of elements MNN allocates for the shape
    /// (which includes channel padding for [`DimensionType::CaffeC4`]) differs from `data.len()`.
    pub fn from_shape_data(shape: impl AsTensorShape, dm_type: DimensionType, data: &[H]) -> Self {
        let shape = shape.as_tensor_shape();
        let size = shape
            .tensor_size()
            .unwrap_or_else(|| panic!("Tensor shape {shape:?} is dynamic or too large"));
        assert_eq!(
            size,
            data.len(),
            "Data length ({}) does not match tensor shape ({:?})",
            data.len(),
            shape
        );
        let mut tensor = Self::new(shape, dm_type);
        let host = tensor.host_mut();
        assert_eq!(
            host.len(),
            data.len(),
            "Tensor layout {dm_type:?} needs {} elements for shape {shape:?} (channel padding) but {} were provided",
            host.len(),
            data.len()
        );
        host.copy_from_slice(data);
        tensor
    }
}

impl<H, M> Tensor<Owned<H>, M>
where
    H: HalideType,
    M: TensorMachine,
{
    /// Create a new tensor with the specified shape and dimension type this allocates the tensor
    /// internally and the data is "owned" by the tensor itself.
    ///
    /// # Panics
    /// Host tensors are allocated immediately, so their shape must not be dynamic (contain
    /// `-1`). Device tensors are only described, not allocated, so any shape is accepted.
    pub fn new(shape: impl AsTensorShape, dm_type: DimensionType) -> Self {
        let shape = shape.as_tensor_shape();
        if M::host() {
            assert!(
                shape.tensor_size().is_some(),
                "Cannot allocate a host tensor with a dynamic or oversized shape {shape:?}"
            );
        }
        let tensor = unsafe {
            if M::device() {
                mnn_sys::Tensor_createDevice(
                    shape.shape.as_ptr(),
                    shape.size,
                    mnn_sys::halide_type_of::<H>(),
                    dm_type.to_mnn_sys(),
                )
            } else {
                mnn_sys::Tensor_createWith(
                    shape.shape.as_ptr(),
                    shape.size,
                    mnn_sys::halide_type_of::<H>(),
                    core::ptr::null_mut(),
                    dm_type.to_mnn_sys(),
                )
            }
        };
        debug_assert!(!tensor.is_null());
        Self {
            tensor,
            destroy_on_drop: true,
            __marker: PhantomData,
        }
    }
}

impl<H> Clone for Tensor<Owned<H>, Host>
where
    H: HalideType + Copy,
{
    fn clone(&self) -> Tensor<Owned<H>, Host> {
        let data = self.host();
        let shape = self.shape();
        let mut out = Tensor::new(shape, self.get_dimension_type());
        out.host_mut().copy_from_slice(data);
        // Cloning / deepCopy  is not supported by mnn currently https://github.com/alibaba/MNN/blob/6b1db4c2eacf8193e6b7a1841acb6e31a1938e82/include/MNN/Tensor.hpp#L131

        out
    }
}

impl<T, H> PartialEq for Tensor<T, Host>
where
    T: TensorType<H = H>,
    H: HalideType + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        if self.size() != other.size() {
            return false;
        }
        let self_data = self.host();
        let other_data = other.host();
        self_data == other_data
    }
}

impl<T, H> Eq for Tensor<T, Host>
where
    T: TensorType<H = H>,
    H: HalideType + Eq,
{
}

/// A tensor shape
#[derive(Clone, Copy)]
#[repr(C)]
pub struct TensorShape {
    pub(crate) shape: [i32; 4],
    pub(crate) size: usize,
}

impl TensorShape {
    /// Number of elements described by the shape (the product of all the extents).
    ///
    /// Returns `None` if the shape is dynamic (contains a negative extent such as `-1`) or
    /// if the product overflows.
    pub fn tensor_size(&self) -> Option<usize> {
        self.shape[..self.size].iter().try_fold(1usize, |acc, &x| {
            let x = usize::try_from(x).ok()?;
            acc.checked_mul(x)
        })
    }
}

impl From<mnn_sys::TensorShape> for TensorShape {
    fn from(value: mnn_sys::TensorShape) -> Self {
        Self {
            shape: value.shape,
            size: value.size,
        }
    }
}

impl From<TensorShape> for mnn_sys::TensorShape {
    fn from(value: TensorShape) -> Self {
        Self {
            shape: value.shape,
            size: value.size,
        }
    }
}

impl core::ops::Deref for TensorShape {
    type Target = [i32];

    fn deref(&self) -> &Self::Target {
        &self.shape[..self.size]
    }
}

impl core::ops::Index<usize> for TensorShape {
    type Output = i32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.shape[..self.size][index]
    }
}

impl core::ops::IndexMut<usize> for TensorShape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.shape[..self.size][index]
    }
}

impl core::ops::DerefMut for TensorShape {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.shape[..self.size]
    }
}

impl core::fmt::Debug for TensorShape {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("TensorShape")
            .field(&&self.shape[..self.size])
            .finish()
    }
}

/// A trait to convert any array-like type to a tensor shape
pub trait AsTensorShape {
    /// Convert the array-like type to a tensor shape
    fn as_tensor_shape(&self) -> TensorShape;
}

impl<T: AsRef<[i32]>> AsTensorShape for T {
    fn as_tensor_shape(&self) -> TensorShape {
        let this = self.as_ref();
        let size = std::cmp::min(this.len(), 4);
        let mut shape = [1; 4];
        shape[..size].copy_from_slice(&this[..size]);
        TensorShape { shape, size }
    }
}

impl AsTensorShape for TensorShape {
    fn as_tensor_shape(&self) -> TensorShape {
        *self
    }
}

#[cfg(test)]
mod as_tensor_shape_tests {
    use super::AsTensorShape;
    macro_rules! shape_test {
        ($t:ty, $kind: expr, $value: expr) => {
            eprintln!("Testing {} with {} shape", stringify!($t), $kind);
            $value.as_tensor_shape();
        };
    }
    #[test]
    fn as_tensor_shape_test_vec() {
        shape_test!(Vec<i32>, "small", vec![1, 2, 3]);
        shape_test!(Vec<i32>, "large", vec![12, 23, 34, 45, 67]);
    }
    #[test]
    fn as_tensor_shape_test_array() {
        shape_test!([i32; 3], "small", [1, 2, 3]);
        shape_test!([i32; 5], "large", [12, 23, 34, 45, 67]);
    }
    #[test]
    fn as_tensor_shape_test_ref() {
        shape_test!(&[i32], "small", &[1, 2, 3]);
        shape_test!(&[i32], "large", &[12, 23, 34, 45, 67]);
    }

    #[test]
    fn tensor_size_handles_dynamic_and_overflow() {
        use super::TensorShape;
        assert_eq!([1, 2, 3].as_tensor_shape().tensor_size(), Some(6));
        assert_eq!([1, -1, 3].as_tensor_shape().tensor_size(), None);
        assert_eq!([i32::MAX; 4].as_tensor_shape().tensor_size(), None);
        // Debug formatting must not panic on dynamic shapes.
        let dynamic = TensorShape {
            shape: [1, 3, -1, 0],
            size: 3,
        };
        assert_eq!(format!("{dynamic:?}"), "TensorShape([1, 3, -1])");
    }
}

#[cfg(test)]
mod tensor_tests {
    #[test]
    #[should_panic]
    fn unsafe_nullptr_tensor() {
        use super::*;
        unsafe {
            super::Tensor::<Owned<i32>, Host>::from_ptr(core::ptr::null_mut());
        }
    }

    /// Verifies that `try_host()` is NOT callable on Device tensors.
    /// Previously this was a soundness hole — calling try_host() on a Device
    /// tensor would dereference a null pointer (segfault). The fix restricts
    /// try_host/host to `Tensor<T, Host>` only.
    #[test]
    fn try_host_not_available_on_device_tensor() {
        use super::*;
        let device_tensor = Tensor::<Owned<f32>, Device>::new([1, 2, 3], DimensionType::Caffe);
        // try_host() should only exist on Host tensors, not Device tensors.
        // Uncommenting the next line must fail to compile:
        // let _ = device_tensor.try_host();
        drop(device_tensor);
    }
}

impl<'a, A> Tensor<View<&'a A>, Host, A>
where
    A: HalideType,
{
    /// Create a tensor that borrows `input` as its storage.
    ///
    /// # Panics
    /// Panics if the shape is dynamic or if the number of elements MNN needs for `shape` in
    /// the `dm_type` layout differs from `input.len()`. Note that [`DimensionType::CaffeC4`]
    /// pads the channel dimension to a multiple of 4, so the slice must already be laid out
    /// (and sized) accordingly.
    pub fn borrowed(shape: impl AsTensorShape, dm_type: DimensionType, input: &'a [A]) -> Self {
        let shape = shape.as_tensor_shape();
        let size = shape
            .tensor_size()
            .unwrap_or_else(|| panic!("Tensor shape {shape:?} is dynamic or too large"));
        assert_eq!(
            size,
            input.len(),
            "Input data length ({}) does not match the tensor shape ({:?})",
            input.len(),
            shape
        );
        let tensor = unsafe {
            mnn_sys::Tensor_createWith(
                shape.shape.as_ptr(),
                shape.size,
                mnn_sys::halide_type_of::<A>(),
                input.as_ptr().cast_mut().cast(),
                dm_type.to_mnn_sys(),
            )
        };
        assert!(!tensor.is_null());
        let output = Self {
            tensor,
            // MNN never frees memory it did not allocate, so destroying the wrapper is safe.
            destroy_on_drop: true,
            __marker: PhantomData,
        };
        // `output` is dropped (and the wrapper destroyed) if this panics.
        assert_eq!(
            output.element_size(),
            input.len(),
            "Tensor layout {dm_type:?} needs {} elements for shape {shape:?} (channel padding) but the slice has {}",
            output.element_size(),
            input.len()
        );
        output
    }
}

impl<'a, A> Tensor<View<&'a mut A>, Host, A>
where
    A: HalideType,
{
    /// Create a tensor that mutably borrows `input` as its storage.
    ///
    /// # Panics
    /// Panics if the shape is dynamic or if the number of elements MNN needs for `shape` in
    /// the `dm_type` layout differs from `input.len()`. Note that [`DimensionType::CaffeC4`]
    /// pads the channel dimension to a multiple of 4, so the slice must already be laid out
    /// (and sized) accordingly.
    pub fn borrowed_mut(
        shape: impl AsTensorShape,
        dm_type: DimensionType,
        input: &'a mut [A],
    ) -> Self {
        let shape = shape.as_tensor_shape();
        let size = shape
            .tensor_size()
            .unwrap_or_else(|| panic!("Tensor shape {shape:?} is dynamic or too large"));
        assert_eq!(
            size,
            input.len(),
            "Input data length ({}) does not match the tensor shape ({:?})",
            input.len(),
            shape
        );
        let len = input.len();
        let tensor = unsafe {
            mnn_sys::Tensor_createWith(
                shape.shape.as_ptr(),
                shape.size,
                mnn_sys::halide_type_of::<A>(),
                input.as_mut_ptr().cast(),
                dm_type.to_mnn_sys(),
            )
        };
        assert!(!tensor.is_null());
        let output = Self {
            tensor,
            // MNN never frees memory it did not allocate, so destroying the wrapper is safe.
            destroy_on_drop: true,
            __marker: PhantomData,
        };
        // `output` is dropped (and the wrapper destroyed) if this panics.
        assert_eq!(
            output.element_size(),
            len,
            "Tensor layout {dm_type:?} needs {} elements for shape {shape:?} (channel padding) but the slice has {}",
            output.element_size(),
            len
        );
        output
    }
}

#[test]
fn test_tensor_borrowed() {
    let shape = [1, 2, 3];
    let data = vec![1, 2, 3, 4, 5, 6];
    let tensor = Tensor::<View<&i32>, Host>::borrowed(shape, DimensionType::Caffe, &data);
    assert_eq!(tensor.shape().as_ref(), shape);
    assert_eq!(tensor.host(), data.as_slice());
}
#[test]
fn test_tensor_borrow_mut() {
    let shape = [1, 2, 3];
    let mut data = vec![1, 2, 3, 4, 5, 6];
    let mut tensor =
        Tensor::<View<&mut i32>, Host>::borrowed_mut(shape, DimensionType::Caffe, &mut data);
    tensor.host_mut().fill(1);
    drop(tensor);
    assert_eq!(data, &[1, 1, 1, 1, 1, 1]);
}

#[test]
#[should_panic]
fn test_tensor_invalid_size() {
    let shape = [300, 400, 500];
    let date = vec![0; 100];
    Tensor::<View<&i32>, Host>::borrowed(shape, DimensionType::default(), &date);
}

#[test]
#[should_panic(expected = "channel padding")]
fn test_tensor_borrowed_rejects_unpadded_c4() {
    // NC4HW4 pads the channel dimension (3) up to 4, so MNN would read 16 elements
    // from a 12 element slice.
    let data = vec![1f32; 12];
    Tensor::<View<&f32>, Host>::borrowed([1, 3, 2, 2], DimensionType::CaffeC4, &data);
}

#[test]
#[should_panic(expected = "channel padding")]
fn test_tensor_borrowed_mut_rejects_unpadded_c4() {
    let mut data = vec![1f32; 12];
    Tensor::<View<&mut f32>, Host>::borrowed_mut([1, 3, 2, 2], DimensionType::CaffeC4, &mut data);
}

#[test]
fn test_tensor_borrowed_padded_c4() {
    // With a channel count that is a multiple of 4 the layouts agree.
    let data = vec![1f32; 16];
    let tensor = Tensor::<View<&f32>, Host>::borrowed([1, 4, 2, 2], DimensionType::CaffeC4, &data);
    assert_eq!(tensor.host().len(), 16);
}

#[test]
#[should_panic(expected = "channel padding")]
fn test_from_shape_data_rejects_unpadded_c4() {
    TensorOwned::<f32, Host>::from_shape_data([1, 3, 2, 2], DimensionType::CaffeC4, &[0f32; 12]);
}

#[test]
#[should_panic(expected = "dynamic")]
fn test_tensor_borrowed_rejects_dynamic_shape() {
    let data = vec![0i32; 6];
    Tensor::<View<&i32>, Host>::borrowed([1, -1, 3], DimensionType::Caffe, &data);
}

#[test]
fn test_view_does_not_destroy_parent() {
    let data = vec![1, 2, 3, 4, 5, 6];
    let owned = TensorOwned::<i32, Host>::from_shape_data([1, 2, 3], DimensionType::Caffe, &data);
    {
        let view = owned.view();
        assert_eq!(view.host(), data.as_slice());
        let reborrowed = view.reborrow();
        assert_eq!(reborrowed.host(), data.as_slice());
    }
    // The parent must still be usable after its views are dropped.
    assert_eq!(owned.host(), data.as_slice());

    let mut owned = owned;
    {
        let mut view = owned.view_mut();
        view.host_mut().fill(7);
        let mut reborrowed = view.reborrow_mut();
        reborrowed.host_mut()[0] = 9;
    }
    assert_eq!(owned.host(), &[9, 7, 7, 7, 7, 7]);
}

#[test]
fn test_tensor_clone() {
    let shape = [1, 2, 3];
    let data = vec![1, 3, 5, 8, 1, 23];
    let mut tensor = Tensor::new(shape, DimensionType::Caffe);
    tensor.host_mut().copy_from_slice(&data);

    let tensor_cloned = tensor.clone();
    assert_eq!(tensor_cloned.host(), data);
}

#[test]
fn test_tensor_creation() {
    let shape = [4, 2, 8];
    let data: &[u8; 64] = b"4756ff762c7a7909a1ec0015836296807602fb2d44e1cac15e057e22b2258040";

    let tensor = {
        let data = data.iter().copied().map(|v| v as f32).collect::<Vec<_>>();
        TensorOwned::<f32, Host>::from_shape_data(shape, DimensionType::Caffe, &data)
    };
    let data = data.iter().copied().map(|v| v as f32).collect::<Vec<_>>();
    assert_eq!(tensor.host(), data);
}
