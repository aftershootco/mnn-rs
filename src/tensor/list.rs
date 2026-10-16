use crate::{AnyTensorRef, Device, TensorRef, prelude::*};
use mnn_sys::HalideType;

/// A list of tensors returned by interpreter output queries.
///
/// This list only hands out shared references to the tensors it describes. Use
/// [`TensorListMut`] (returned by [`Interpreter::inputs`](crate::Interpreter::inputs))
/// when you need to mutate the tensors.
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
    pub fn iter<'tl>(&'tl self) -> TensorListIter<'t, 'tl> {
        TensorListIter {
            tensor_list: self,
            idx: 0,
        }
    }
}

impl<'t, 'tl> IntoIterator for &'tl TensorList<'t> {
    type Item = TensorInfo<'t, 'tl>;
    type IntoIter = TensorListIter<'t, 'tl>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// A list of tensors returned by interpreter input queries.
///
/// This list is created from a `&mut Session` and therefore has exclusive access to the
/// tensors it describes, which is what allows it to hand out `&mut TensorRef`s.
/// It dereferences to [`TensorList`] for all the read-only operations.
#[repr(transparent)]
pub struct TensorListMut<'t> {
    inner: TensorList<'t>,
}

impl core::fmt::Debug for TensorListMut<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> core::fmt::Result {
        self.inner.fmt(f)
    }
}

impl<'t> core::ops::Deref for TensorListMut<'t> {
    type Target = TensorList<'t>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'t> TensorListMut<'t> {
    pub(crate) fn from_ptr(inner: *const mnn_sys::TensorInfoArray) -> Self {
        Self {
            inner: TensorList::from_ptr(inner),
        }
    }

    /// Get a mutable reference to the tensor at the given index
    pub fn get_mut(&mut self, index: usize) -> Option<TensorInfoMut<'t, '_>> {
        if index >= self.inner.size() {
            None
        } else {
            let gtinfo = unsafe { mnn_sys::getTensorInfoArray(self.inner.inner, index) };
            if !gtinfo.is_null() {
                Some(TensorInfoMut {
                    tensor_info: gtinfo,
                    __marker: PhantomData,
                })
            } else {
                None
            }
        }
    }

    /// Get a mutable iterator over the tensor list
    pub fn iter_mut<'tl>(&'tl mut self) -> TensorListIterMut<'t, 'tl> {
        TensorListIterMut {
            tensor_list: self,
            idx: 0,
        }
    }
}

impl<'t, 'tl> IntoIterator for &'tl TensorListMut<'t> {
    type Item = TensorInfo<'t, 'tl>;
    type IntoIter = TensorListIter<'t, 'tl>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

impl<'t, 'tl> IntoIterator for &'tl mut TensorListMut<'t> {
    type Item = TensorInfoMut<'t, 'tl>;
    type IntoIter = TensorListIterMut<'t, 'tl>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// Iterator over tensor list
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

/// Mutable iterator over tensor list
pub struct TensorListIterMut<'t, 'tl> {
    tensor_list: &'tl mut TensorListMut<'t>,
    idx: usize,
}

impl<'t, 'tl> Iterator for TensorListIterMut<'t, 'tl> {
    type Item = TensorInfoMut<'t, 'tl>;
    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.idx;
        self.idx += 1;
        let out = self.tensor_list.get_mut(idx);
        // SAFETY: Every index is visited at most once, so each `TensorInfoMut` handed out by
        // this iterator refers to a distinct entry of the list. The iterator holds the list
        // exclusively for `'tl`, so extending the borrow from the `&mut self` of `get_mut`
        // to `'tl` cannot create overlapping mutable access. This is the same reasoning the
        // standard library uses for `slice::IterMut`.
        unsafe {
            core::mem::transmute::<Option<TensorInfoMut<'t, '_>>, Option<TensorInfoMut<'t, 'tl>>>(
                out,
            )
        }
    }
}

/// Information about a tensor in a tensor list
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

/// Mutable information about a tensor in a tensor list
///
/// Each `TensorInfoMut` represents exclusive access to one entry of a [`TensorListMut`] for
/// the lifetime `'tl`.
#[repr(transparent)]
pub struct TensorInfoMut<'t, 'tl> {
    pub(crate) tensor_info: *mut mnn_sys::TensorInfo,
    pub(crate) __marker: PhantomData<&'tl mut TensorList<'t>>,
}

impl core::fmt::Debug for TensorInfoMut<'_, '_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let tensor = self.raw_tensor();
        let shape = tensor.shape();
        f.debug_struct("TensorInfoMut")
            .field("name", &self.name())
            .field("tensor", &shape)
            .finish()
    }
}

/// Type and shape checks shared by all the typed accessors
fn check_tensor<H: HalideType>(tensor: &AnyTensorRef) -> Result<()> {
    ensure!(!tensor.is_dynamic_unsized(), ErrorKind::DynamicTensorError);
    ensure!(
        tensor.is_type_of::<H>(),
        ErrorKind::HalideTypeMismatch {
            got: std::any::type_name::<H>(),
        }
    );
    Ok(())
}

impl<'t, 'tl> TensorInfoMut<'t, 'tl> {
    fn tensor_ptr(&self) -> *mut mnn_sys::Tensor {
        debug_assert!(!self.tensor_info.is_null());
        let tensor = unsafe { (*self.tensor_info).tensor };
        debug_assert!(!tensor.is_null());
        tensor.cast()
    }

    /// Get the name of the tensor
    pub fn name(&self) -> &'tl str {
        debug_assert!(!self.tensor_info.is_null());
        unsafe { (*self.tensor_info).name.to_cstr() }
            .to_str()
            .expect("Tensor name is not utf-8")
    }

    /// Get the tensor with the specified type
    pub fn tensor<H: HalideType>(&self) -> Result<&TensorRef<H, Device>> {
        check_tensor::<H>(self.raw_tensor())?;
        Ok(unsafe { crate::tensor::from_raw_parts(self.tensor_ptr()) })
    }

    /// Get a mutable reference to the tensor with the specified type
    ///
    /// The returned reference borrows `self` mutably, use [`Self::into_tensor_mut`] if you
    /// need a reference that lives as long as the list borrow.
    pub fn tensor_mut<H: HalideType>(&mut self) -> Result<&mut TensorRef<H, Device>> {
        check_tensor::<H>(self.raw_tensor())?;
        Ok(unsafe { crate::tensor::from_raw_parts_mut(self.tensor_ptr()) })
    }

    /// Consume this entry and get a mutable reference to the tensor that lives as long as
    /// the borrow of the [`TensorListMut`] this entry came from.
    pub fn into_tensor_mut<H: HalideType>(self) -> Result<&'tl mut TensorRef<H, Device>> {
        check_tensor::<H>(self.raw_tensor())?;
        // SAFETY: `self` represents exclusive access to this entry for `'tl` and is consumed
        // here, so no other reference to the tensor can be created through it.
        Ok(unsafe { crate::tensor::from_raw_parts_mut(self.tensor_ptr()) })
    }

    /// # Safety
    /// The shape is not checked so it's marked unsafe since futher calls to interpreter might be **unsafe** with this
    pub unsafe fn tensor_unresized<H: HalideType>(&mut self) -> Result<&mut TensorRef<H, Device>> {
        let tensor = self.raw_tensor();
        ensure!(
            tensor.is_type_of::<H>(),
            ErrorKind::HalideTypeMismatch {
                got: std::any::type_name::<H>(),
            }
        );
        Ok(unsafe { crate::tensor::from_raw_parts_mut(self.tensor_ptr()) })
    }

    /// This function return's the raw tensor without any sort of type-checking or shape-checking
    pub fn raw_tensor(&self) -> &AnyTensorRef {
        unsafe { AnyTensorRef::from_ptr(self.tensor_ptr()) }
    }

    /// This function return's the raw tensor mutably without any sort of type-checking or shape-checking
    pub fn raw_tensor_mut(&mut self) -> &mut AnyTensorRef {
        unsafe { AnyTensorRef::from_mut_ptr(self.tensor_ptr()) }
    }
}

impl<'t, 'tl> TensorInfo<'t, 'tl> {
    fn tensor_ptr(&self) -> *mut mnn_sys::Tensor {
        debug_assert!(!self.tensor_info.is_null());
        let tensor = unsafe { (*self.tensor_info).tensor };
        debug_assert!(!tensor.is_null());
        tensor.cast()
    }

    /// Get the name of the tensor
    pub fn name(&self) -> &'tl str {
        debug_assert!(!self.tensor_info.is_null());
        unsafe { (*self.tensor_info).name.to_cstr() }
            .to_str()
            .expect("Tensor name is not utf-8")
    }

    /// Get the tensor with the specified type
    pub fn tensor<H: HalideType>(&self) -> Result<&'tl TensorRef<H, Device>> {
        check_tensor::<H>(self.raw_tensor())?;
        Ok(unsafe { crate::tensor::from_raw_parts(self.tensor_ptr()) })
    }

    /// This function return's the raw tensor without any sort of type-checking or shape-checking
    pub fn raw_tensor(&self) -> &'tl AnyTensorRef {
        unsafe { AnyTensorRef::from_ptr(self.tensor_ptr()) }
    }
}
