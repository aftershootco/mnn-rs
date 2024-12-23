#![deny(missing_docs)]
use crate::{prelude::*, Device, RawTensor, RefMut, Tensor};
use mnn_sys::HalideType;

#[repr(transparent)]
pub struct TensorList<'t> {
    pub(crate) inner: *const mnn_sys::TensorInfoArray,
    pub(crate) __marker: PhantomData<&'t ()>,
}

impl core::fmt::Debug for TensorList<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl Drop for TensorList<'_> {
    fn drop(&mut self) {
        unsafe { mnn_sys::destroyTensorInfoArray(self.inner.cast_mut()) }
    }
}

impl<'t> TensorList<'t> {
    pub(crate) fn from_ptr(inner: *const mnn_sys::TensorInfoArray) -> Self {
        Self {
            inner,
            __marker: PhantomData,
        }
    }

    /// Returns the size of the tensor list
    pub fn size(&self) -> usize {
        unsafe { (*self.inner).size }
    }

    /// Get the tensor at the given index
    pub fn get(&self, index: usize) -> Option<TensorInfo<'t, '_>> {
        if index >= self.size() {
            None
        } else {
            let gtinfo = unsafe { mnn_sys::getTensorInfoArray(self.inner, index) };
            if !gtinfo.is_null() {
                Some(TensorInfo {
                    tensor_info: gtinfo,
                    __marker: PhantomData,
                })
            } else {
                None
            }
        }
    }

    /// Get an iterator over the tensor list
    pub fn iter(&self) -> TensorListIter {
        TensorListIter {
            tensor_list: self,
            idx: 0,
        }
    }
}

impl<'t, 'tl: 't> IntoIterator for &'tl TensorList<'t> {
    type Item = TensorInfo<'t, 'tl>;
    type IntoIter = TensorListIter<'t, 'tl>;

    fn into_iter(self) -> Self::IntoIter {
        TensorListIter {
            tensor_list: self,
            idx: 0,
        }
    }
}

pub struct TensorListIter<'t, 'tl> {
    tensor_list: &'tl TensorList<'t>,
    idx: usize,
}
impl<'t, 'tl> Iterator for TensorListIter<'t, 'tl> {
    type Item = TensorInfo<'t, 'tl>;
    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.idx;
        self.idx += 1;
        self.tensor_list.get(idx)
    }
}

#[repr(transparent)]
pub struct TensorInfo<'t, 'tl> {
    pub(crate) tensor_info: *mut mnn_sys::TensorInfo,
    pub(crate) __marker: PhantomData<&'tl TensorList<'t>>,
}

impl core::fmt::Debug for TensorInfo<'_, '_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let tensor = self.raw_tensor();
        let shape = tensor.shape();
        f.debug_struct("TensorInfo")
            .field("name", &self.name())
            .field("tensor", &shape)
            .finish()
    }
}

impl<'t, 'tl> TensorInfo<'t, 'tl> {
    pub fn name(&self) -> &'tl str {
        debug_assert!(!self.tensor_info.is_null());
        unsafe { (*self.tensor_info).name.to_cstr() }
            .to_str()
            .expect("Tensor name is not utf-8")
    }

    pub fn tensor<H: HalideType>(&self) -> Result<Tensor<RefMut<'t, Device<H>>>> {
        debug_assert!(!self.tensor_info.is_null());
        unsafe { debug_assert!(!(*self.tensor_info).tensor.is_null()) };
        let tensor = unsafe { Tensor::from_ptr((*self.tensor_info).tensor.cast()) };
        let shape = tensor.shape();
        ensure!(!shape.as_ref().contains(&-1), ErrorKind::DynamicTensorError);
        ensure!(
            tensor.is_type_of::<H>(),
            ErrorKind::HalideTypeMismatch {
                got: std::any::type_name::<H>(),
            }
        );
        Ok(tensor)
    }

    /// # Safety
    /// The shape is not checked so it's marked unsafe since futher calls to interpreter might be **unsafe** with this
    pub unsafe fn tensor_unresized<H: HalideType>(&self) -> Result<Tensor<RefMut<'t, Device<H>>>> {
        debug_assert!(!self.tensor_info.is_null());
        unsafe { debug_assert!(!(*self.tensor_info).tensor.is_null()) };
        let tensor = unsafe { Tensor::from_ptr((*self.tensor_info).tensor.cast()) };
        ensure!(
            tensor.is_type_of::<H>(),
            ErrorKind::HalideTypeMismatch {
                got: std::any::type_name::<H>(),
            }
        );
        Ok(tensor)
    }

    /// This function return's the raw tensor without any sort of type-checking or shape-checking
    pub fn raw_tensor(&self) -> RawTensor<'t> {
        debug_assert!(!self.tensor_info.is_null());
        unsafe { debug_assert!(!(*self.tensor_info).tensor.is_null()) };
        RawTensor::from_ptr(unsafe { (*self.tensor_info).tensor.cast() })
    }
}
