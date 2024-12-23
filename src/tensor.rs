use crate::prelude::*;
use core::marker::PhantomData;
use mnn_sys::*;
pub(crate) mod list;
mod raw;
pub use raw::RawTensor;

use mnn_sys::HalideType;

mod seal {
    pub trait Sealed {}
}
macro_rules! seal {
        ($($name:ty),*) => {
            $(
                impl<T> seal::Sealed for $name {}
            )*
        };
    }
seal!(Host<T>, Device<T>, Ref<'_, T>, RefMut<'_, T>);

/// A trait to represent the type of a tensor
pub trait TensorType: seal::Sealed {
    /// The halide type of the tensor
    type H;
    /// Check if the tensor is owned
    fn owned() -> bool;
    /// Check if the tensor is borrowed
    fn borrowed() -> bool {
        !Self::owned()
    }
    /// Check if the tensor is allocated in the host
    fn host() -> bool;
    /// Check if the tensor is allocated in the device
    fn device() -> bool {
        !Self::host()
    }
}
/// A tensor that is owned
pub trait OwnedTensorType: TensorType {}
/// A tensor that is borrowed
pub trait RefTensorType: TensorType {}
/// A tensor that is allocated in the cpu / host platform
pub trait HostTensorType: TensorType {}
/// A tensor that is allocated in the device / gpu platform
pub trait DeviceTensorType: TensorType {}
/// A tensor that is mutable
pub trait MutableTensorType: TensorType {}

impl<H: HalideType> TensorType for Host<H> {
    type H = H;
    fn owned() -> bool {
        true
    }
    fn host() -> bool {
        true
    }
}
impl<H: HalideType> TensorType for Device<H> {
    type H = H;
    fn owned() -> bool {
        true
    }
    fn host() -> bool {
        false
    }
}

impl<T: TensorType> TensorType for Ref<'_, T> {
    type H = T::H;
    fn owned() -> bool {
        false
    }
    fn host() -> bool {
        T::host()
    }
}

impl<T: TensorType> TensorType for RefMut<'_, T> {
    type H = T::H;
    fn owned() -> bool {
        false
    }
    fn host() -> bool {
        T::host()
    }
}

impl<H: HalideType> DeviceTensorType for Device<H> {}
impl<H: HalideType> HostTensorType for Host<H> {}
impl<H: HalideType> OwnedTensorType for Device<H> {}
impl<H: HalideType> OwnedTensorType for Host<H> {}
impl<T: DeviceTensorType> DeviceTensorType for Ref<'_, T> {}
impl<T: DeviceTensorType> DeviceTensorType for RefMut<'_, T> {}
impl<T: HostTensorType> HostTensorType for Ref<'_, T> {}
impl<T: HostTensorType> HostTensorType for RefMut<'_, T> {}
impl<T: OwnedTensorType> MutableTensorType for T {}
impl<T: TensorType> MutableTensorType for RefMut<'_, T> {}
impl<T: TensorType> RefTensorType for Ref<'_, T> {}
impl<T: TensorType> RefTensorType for RefMut<'_, T> {}

/// A tensor that is owned by the cpu / host platform
pub struct Host<T = f32> {
    pub(crate) __marker: PhantomData<T>,
}
/// A tensor that is owned by the device / gpu platform
pub struct Device<T = f32> {
    pub(crate) __marker: PhantomData<T>,
}
/// A reference to a any tensor
pub struct Ref<'t, T> {
    pub(crate) __marker: PhantomData<&'t [T]>,
}

/// A mutable reference to a any tensor
pub struct RefMut<'t, T> {
    pub(crate) __marker: PhantomData<&'t mut [T]>,
}

/// A generic tensor that can of host / device / owned / borrowed
pub struct Tensor<T: TensorType> {
    pub(crate) tensor: *mut mnn_sys::Tensor,
    __marker: PhantomData<T>,
}

impl<T: TensorType> Drop for Tensor<T> {
    fn drop(&mut self) {
        if T::owned() {
            unsafe {
                mnn_sys::Tensor_destroy(self.tensor);
            }
        }
    }
}

impl<H: HalideType> Tensor<Host<H>> {
    /// Get's a reference to an owned host tensor
    pub fn as_ref(&self) -> Tensor<Ref<'_, Host<H>>> {
        Tensor {
            tensor: self.tensor,
            __marker: PhantomData,
        }
    }
}

impl<H: HalideType> Tensor<Device<H>> {
    /// Get's a reference to an owned device tensor
    pub fn as_ref(&self) -> Tensor<Ref<'_, Device<H>>> {
        Tensor {
            tensor: self.tensor,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DimensionType {
    /// Caffe style dimensions or NCHW
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

impl<T: TensorType> Tensor<T>
where
    T::H: HalideType,
{
    /// This function constructs a Tensor type from a raw pointer
    ///# Safety
    /// Since this constructs a Tensor from a raw pointer we have no way to guarantee that it's a
    /// valid tensor or it's lifetime
    pub unsafe fn from_ptr(tensor: *mut mnn_sys::Tensor) -> Self {
        assert!(!tensor.is_null());
        Self {
            tensor,
            __marker: PhantomData,
        }
    }
    /// Copies the data from a host tensor to the self tensor
    pub fn copy_from_host_tensor(&mut self, tensor: &Tensor<Host<T::H>>) -> Result<()> {
        let ret = unsafe { Tensor_copyFromHostTensor(self.tensor, tensor.tensor) };
        crate::ensure!(ret != 0, ErrorKind::TensorCopyFailed(ret));
        Ok(())
    }

    /// Copies the data from the self tensor to a host tensor
    pub fn copy_to_host_tensor(&self, tensor: &mut Tensor<Host<T::H>>) -> Result<()> {
        let ret = unsafe { Tensor_copyToHostTensor(self.tensor, tensor.tensor) };
        crate::ensure!(ret != 0, ErrorKind::TensorCopyFailed(ret));
        Ok(())
    }

    /// Get the device id of the tensor
    pub fn device_id(&self) -> u64 {
        unsafe { Tensor_deviceId(self.tensor) }
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> TensorShape {
        unsafe { Tensor_shape(self.tensor) }.into()
    }

    /// Get the dimensions of the tensor
    pub fn dimensions(&self) -> usize {
        unsafe { Tensor_dimensions(self.tensor) as usize }
    }

    /// Get the width of the tensor
    pub fn width(&self) -> u32 {
        unsafe { Tensor_width(self.tensor) as u32 }
    }

    /// Get the height of the tensor
    pub fn height(&self) -> u32 {
        unsafe { Tensor_height(self.tensor) as u32 }
    }

    /// Get the channel size of the tensor
    pub fn channel(&self) -> u32 {
        unsafe { Tensor_channel(self.tensor) as u32 }
    }

    /// Get the batch size of the tensor
    pub fn batch(&self) -> u32 {
        unsafe { Tensor_batch(self.tensor) as u32 }
    }

    /// Get the size of the tensor when counted by bytes
    pub fn size(&self) -> usize {
        unsafe { Tensor_usize(self.tensor) }
    }

    /// Get the size of the tensor when counted by elements
    pub fn element_size(&self) -> usize {
        unsafe { Tensor_elementSize(self.tensor) as usize }
    }

    /// Print the shape of the tensor
    pub fn print_shape(&self) {
        unsafe {
            Tensor_printShape(self.tensor);
        }
    }

    /// Print the tensor
    pub fn print(&self) {
        unsafe {
            Tensor_print(self.tensor);
        }
    }

    /// Check if the tensor is dynamic and needs resizing
    pub fn is_dynamic_unsized(&self) -> bool {
        self.shape().as_ref().contains(&-1)
    }

    /// DO not use this function directly
    /// # Safety
    /// This is just provided as a 1:1 compat mostly for possible later use
    pub unsafe fn halide_buffer(&self) -> *const halide_buffer_t {
        Tensor_buffer(self.tensor)
    }

    /// Do not use this function directly
    /// # Safety
    /// This is just provided as a 1:1 compat mostly for possible later use
    pub unsafe fn halide_buffer_mut(&self) -> *mut halide_buffer_t {
        Tensor_buffer_mut(self.tensor)
    }

    /// Get the dimension type of the tensor
    pub fn get_dimension_type(&self) -> DimensionType {
        debug_assert!(!self.tensor.is_null());
        From::from(unsafe { Tensor_getDimensionType(self.tensor) })
    }

    /// Get the data type of the tensor
    pub fn get_type(&self) -> mnn_sys::halide_type_t {
        unsafe { Tensor_getType(self.tensor) }
    }

    /// Check if the tensor is of the specified data type
    pub fn is_type_of<H: HalideType>(&self) -> bool {
        let htc = halide_type_of::<H>();
        unsafe { Tensor_isTypeOf(self.tensor, htc) }
    }

    /// # Safety
    /// This is very unsafe do not use this unless you know what you are doing
    pub unsafe fn into_raw(self) -> RawTensor<'static> {
        let out = RawTensor {
            inner: self.tensor,
            __marker: PhantomData,
        };
        core::mem::forget(self);
        out
    }
}
impl<T: MutableTensorType> Tensor<T>
where
    T::H: HalideType,
{
    /// Fill the tensor with the specified value
    pub fn fill(&mut self, value: T::H)
    where
        T::H: Copy,
    {
        if T::host() {
            let size = self.element_size();
            assert!(self.is_type_of::<T::H>());
            let result: &mut [T::H] = unsafe {
                let data = mnn_sys::Tensor_host_mut(self.tensor).cast();
                core::slice::from_raw_parts_mut(data, size)
            };
            result.fill(value);
        } else if T::device() {
            let shape = self.shape();
            let dm_type = self.get_dimension_type();
            let mut host = Tensor::new(shape, dm_type);
            host.fill(value);
            self.copy_from_host_tensor(&host)
                .expect("Failed to copy data from host tensor");
        } else {
            unreachable!()
        }
    }
}

impl<T: HostTensorType> Tensor<T>
where
    T::H: HalideType,
{
    /// Try to map the device tensor to the host memory and get the slice
    pub fn try_host(&self) -> Result<&[T::H]> {
        let size = self.element_size();
        ensure!(
            self.is_type_of::<T::H>(),
            ErrorKind::HalideTypeMismatch {
                got: std::any::type_name::<T::H>(),
            }
        );
        let result = unsafe {
            let data = mnn_sys::Tensor_host(self.tensor).cast();
            core::slice::from_raw_parts(data, size)
        };
        Ok(result)
    }

    /// Try to map the device tensor to the host memory and get the mutable slice
    pub fn try_host_mut(&mut self) -> Result<&mut [T::H]> {
        let size = self.element_size();
        ensure!(
            self.is_type_of::<T::H>(),
            ErrorKind::HalideTypeMismatch {
                got: std::any::type_name::<T::H>(),
            }
        );

        let result = unsafe {
            let data: *mut T::H = mnn_sys::Tensor_host_mut(self.tensor).cast();
            debug_assert!(!data.is_null());
            core::slice::from_raw_parts_mut(data, size)
        };
        Ok(result)
    }

    /// Get the host memory slice of the tensor
    pub fn host(&self) -> &[T::H] {
        self.try_host().expect("Failed to get tensor host")
    }

    /// Get the mutable host memory slice of the tensor
    pub fn host_mut(&mut self) -> &mut [T::H] {
        self.try_host_mut().expect("Failed to get tensor host_mut")
    }
}

impl<T: DeviceTensorType> Tensor<T>
where
    T::H: HalideType,
{
    /// Try to wait for the device tensor to finish processing
    pub fn wait(&self, map_type: MapType, finish: bool) {
        unsafe {
            Tensor_wait(self.tensor, map_type, finish as i32);
        }
    }

    /// Create a host tensor from the device tensor with same dimensions and data type and
    /// optionally copy the data from the device tensor
    pub fn create_host_tensor_from_device(&self, copy_data: bool) -> Tensor<Host<T::H>> {
        let shape = self.shape();
        let dm_type = self.get_dimension_type();
        let mut out = Tensor::new(shape, dm_type);

        if copy_data {
            self.copy_to_host_tensor(&mut out)
                .expect("Failed to copy data from device tensor");
        }
        out
    }
}

impl<T: OwnedTensorType> Tensor<T>
where
    T::H: HalideType,
{
    /// Create a new tensor with the specified shape and dimension type
    pub fn new(shape: impl AsTensorShape, dm_type: DimensionType) -> Self {
        let shape = shape.as_tensor_shape();
        let tensor = unsafe {
            if T::device() {
                Tensor_createDevice(
                    shape.shape.as_ptr(),
                    shape.size,
                    halide_type_of::<T::H>(),
                    dm_type.to_mnn_sys(),
                )
            } else {
                Tensor_createWith(
                    shape.shape.as_ptr(),
                    shape.size,
                    halide_type_of::<T::H>(),
                    core::ptr::null_mut(),
                    DimensionType::Caffe.to_mnn_sys(),
                )
            }
        };
        debug_assert!(!tensor.is_null());
        Self {
            tensor,
            __marker: PhantomData,
        }
    }
}

impl<T: OwnedTensorType> Clone for Tensor<T>
where
    T::H: HalideType,
{
    fn clone(&self) -> Tensor<T> {
        let tensor_ptr = unsafe { Tensor_clone(self.tensor) };
        Self {
            tensor: tensor_ptr,
            __marker: PhantomData,
        }
    }
}

/// A tensor shape
#[derive(Clone, Copy)]
#[repr(C)]
pub struct TensorShape {
    pub(crate) shape: [i32; 4],
    pub(crate) size: usize,
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
        write!(f, "{:?}", &self.shape[..self.size])
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
}

#[cfg(test)]
mod tensor_tests {
    #[test]
    #[should_panic]
    fn unsafe_nullptr_tensor() {
        unsafe {
            super::Tensor::<super::Host<i32>>::from_ptr(core::ptr::null_mut());
        }
    }
}

impl<T: HostTensorType + RefTensorType> Tensor<T>
where
    T::H: HalideType,
{
    /// Try to create a ref tensor from any array-like type
    pub fn borrowed(shape: impl AsTensorShape, input: impl AsRef<[T::H]>) -> Self {
        let shape = shape.as_tensor_shape();
        let input = input.as_ref();
        let tensor = unsafe {
            Tensor_createWith(
                shape.shape.as_ptr(),
                shape.size,
                halide_type_of::<T::H>(),
                input.as_ptr().cast_mut().cast(),
                DimensionType::Caffe.to_mnn_sys(),
            )
        };
        debug_assert!(!tensor.is_null());
        Self {
            tensor,
            __marker: PhantomData,
        }
    }

    /// Try to create a mutable ref tensor from any array-like type
    pub fn borrowed_mut(shape: impl AsTensorShape, mut input: impl AsMut<[T::H]>) -> Self {
        let shape = shape.as_tensor_shape();
        let input = input.as_mut();
        let tensor = unsafe {
            Tensor_createWith(
                shape.shape.as_ptr(),
                shape.size,
                halide_type_of::<T::H>(),
                input.as_mut_ptr().cast(),
                DimensionType::Caffe.to_mnn_sys(),
            )
        };
        debug_assert!(!tensor.is_null());
        Self {
            tensor,
            __marker: PhantomData,
        }
    }
}

#[test]
fn test_tensor_borrowed() {
    let shape = [1, 2, 3];
    let data = vec![1, 2, 3, 4, 5, 6];
    let tensor = Tensor::<Ref<Host<i32>>>::borrowed(&shape, &data);
    assert_eq!(tensor.shape().as_ref(), shape);
    assert_eq!(tensor.host(), data.as_slice());
}

#[test]
fn test_tensor_borrow_mut() {
    let shape = [1, 2, 3];
    let mut data = vec![1, 2, 3, 4, 5, 6];
    let mut tensor = Tensor::<RefMut<Host<i32>>>::borrowed_mut(&shape, &mut data);
    tensor.host_mut().fill(1);
    assert_eq!(data, &[1, 1, 1, 1, 1, 1]);
}
