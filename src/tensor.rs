use crate::prelude::*;
use core::marker::PhantomData;

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
seal!(Host<T>, Device<T>, Ref<'_, T>);

pub trait TensorType: seal::Sealed {
    type H: HalideType;
    fn owned() -> bool;
    fn borrowed() -> bool {
        !Self::owned()
    }
    fn host() -> bool;
    fn device() -> bool {
        !Self::host()
    }
}
pub trait OwnedTensorType: TensorType {}
pub trait RefTensorType: TensorType {}
pub trait HostTensorType: TensorType {}
pub trait DeviceTensorType: TensorType {}

impl<H: HalideType> TensorType for Host<H> {
    type H = H;
    fn owned() -> bool {
        true
    }
    fn host() -> bool {
        true
    }
}
impl<H: HalideType> OwnedTensorType for Host<H> {}
impl<H: HalideType> HostTensorType for Host<H> {}
impl<H: HalideType> TensorType for Device<H> {
    type H = H;
    fn owned() -> bool {
        true
    }
    fn host() -> bool {
        false
    }
}
impl<H: HalideType> OwnedTensorType for Device<H> {}
impl<H: HalideType> DeviceTensorType for Device<H> {}

impl<T: TensorType> TensorType for Ref<'_, T> {
    type H = T::H;
    fn owned() -> bool {
        false
    }
    fn host() -> bool {
        T::host()
    }
}
impl<T: HostTensorType> HostTensorType for Ref<'_, T> {}
impl<T: DeviceTensorType> DeviceTensorType for Ref<'_, T> {}

pub struct Host<T = f32> {
    pub(crate) __marker: PhantomData<T>,
}
pub struct Device<T = f32> {
    pub(crate) __marker: PhantomData<T>,
}
pub struct Ref<'t, T> {
    pub(crate) __marker: PhantomData<(&'t (), T)>,
}

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
    pub fn as_ref(&self) -> Tensor<Ref<'_, Host<H>>> {
        Tensor {
            tensor: self.tensor,
            __marker: PhantomData,
        }
    }
}

impl<H: HalideType> Tensor<Device<H>> {
    pub fn as_ref(&self) -> Tensor<Ref<'_, Device<H>>> {
        Tensor {
            tensor: self.tensor,
            __marker: PhantomData,
        }
    }
}

use mnn_sys::*;
impl<T: TensorType> Tensor<T> {
    pub unsafe fn from_ptr(tensor: *mut mnn_sys::Tensor) -> Self {
        Self {
            tensor,
            __marker: PhantomData,
        }
    }
    pub fn copy_from_host_tensor(&mut self, tensor: &Tensor<Host<T::H>>) -> Result<()> {
        let ret = unsafe { Tensor_copyFromHostTensor(self.tensor, tensor.tensor) };
        crate::ensure!(ret != 0, ErrorKind::TensorCopyFailed);
        Ok(())
    }

    pub fn copy_to_host_tensor(&self, tensor: &mut Tensor<Host<T::H>>) -> Result<()> {
        let ret = unsafe { Tensor_copyToHostTensor(self.tensor, tensor.tensor) };
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

    pub fn width(&self) -> u32 {
        unsafe { Tensor_width(self.tensor) as u32 }
    }

    pub fn height(&self) -> u32 {
        unsafe { Tensor_height(self.tensor) as u32 }
    }

    pub fn channel(&self) -> u32 {
        unsafe { Tensor_channel(self.tensor) as u32 }
    }

    pub fn batch(&self) -> u32 {
        unsafe { Tensor_batch(self.tensor) as u32 }
    }

    pub fn size(&self) -> usize {
        unsafe { Tensor_usize(self.tensor) }
    }

    pub fn element_size(&self) -> usize {
        unsafe { Tensor_elementSize(self.tensor) as usize }
    }

    pub fn print_shape(&self) {
        unsafe {
            Tensor_printShape(self.tensor);
        }
    }

    pub fn print(&self) {
        unsafe {
            Tensor_print(self.tensor);
        }
    }

    /// DO not use this function directly
    pub unsafe fn halide_buffer(&self) -> *const halide_buffer_t {
        Tensor_buffer(self.tensor)
    }

    /// DO not use this function directly
    pub unsafe fn halide_buffer_mut(&self) -> *mut halide_buffer_t {
        Tensor_buffer_mut(self.tensor)
    }

    pub fn get_diemension_type(&self) -> mnn_sys::DimensionType {
        unsafe { Tensor_getDimensionType(self.tensor) }
    }

    pub fn get_type(&self) -> mnn_sys::halide_type_c {
        let type_ = unsafe { Tensor_getType(self.tensor) };
        type_
    }

    pub fn is_type_of<H: HalideType>(&self) -> bool {
        let htc = halide_type_of::<H>();
        let htc = halide_type_c {
            code: unsafe { halide_type_code_t::from_u32(htc.code as u32) },
            bits: htc.bits,
            lanes: htc.lanes,
        };
        unsafe { Tensor_isTypeOf(self.tensor, htc) }
    }
}

impl<T: HostTensorType> Tensor<T> {
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

    pub fn host(&self) -> &[T::H] {
        self.try_host().expect("Failed to get tensor host")
    }

    pub fn host_mut(&mut self) -> &mut [T::H] {
        self.try_host_mut().expect("Failed to get tensor host_mut")
    }
}

impl<T: DeviceTensorType> Tensor<T> {
    pub fn wait(&self, map_type: MapType, finish: bool) {
        unsafe {
            Tensor_wait(self.tensor, map_type, finish as i32);
        }
    }
    pub fn create_host_tensor_from_device(&self, copy_data: bool) -> Tensor<Host<T::H>> {
        let tensor = unsafe { Tensor_createHostTensorFromDevice(self.tensor, copy_data as i32) };
        debug_assert!(!tensor.is_null());

        unsafe { Tensor::from_ptr(tensor) }
    }
}

impl<T: OwnedTensorType> Clone for Tensor<T> {
    fn clone(&self) -> Tensor<T> {
        let tensor_ptr = unsafe { Tensor_clone(self.tensor) };
        Self {
            tensor: tensor_ptr,
            __marker: PhantomData,
        }
    }
}
