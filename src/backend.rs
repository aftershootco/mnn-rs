use mnn_sys::*;
pub use mnn_sys::{MemoryMode, PowerMode, PrecisionMode};

#[repr(transparent)]
pub struct BackendConfig {
    pub inner: *mut MNNBackendConfig,
    __marker: core::marker::PhantomData<()>,
}

impl BackendConfig {
    pub fn as_ptr_mut(&self) -> *mut MNNBackendConfig {
        self.inner
    }
    pub fn new() -> Self {
        unsafe {
            let inner = mnnbc_create();
            Self {
                inner,
                __marker: core::marker::PhantomData,
            }
        }
    }

    pub fn set_memory_mode(&mut self, mode: MemoryMode) {
        unsafe {
            mnn_sys::mnnbc_set_memory_mode(self.inner, mode);
        }
    }

    pub fn set_power_mode(&mut self, mode: PowerMode) {
        unsafe {
            mnn_sys::mnnbc_set_power_mode(self.inner, mode);
        }
    }

    pub fn set_precision_mode(&mut self, mode: PrecisionMode) {
        unsafe {
            mnn_sys::mnnbc_set_precision_mode(self.inner, mode);
        }
    }

    pub fn set_flags(&mut self, flags: usize) {
        unsafe {
            mnn_sys::mnnbc_set_flags(self.inner, flags);
        }
    }

    pub unsafe fn set_shared_context(&mut self, shared_context: *mut core::ffi::c_void) {
        unsafe {
            mnn_sys::mnnbc_set_shared_context(self.inner, shared_context);
        }
    }
}
