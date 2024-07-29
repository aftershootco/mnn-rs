use anyhow::*;
use mnn_sys::*;
pub struct ScheduleConfig {
    pub(crate) inner: *mut MNNScheduleConfig,
    pub(crate) __marker: core::marker::PhantomData<()>,
}

impl ScheduleConfig {
    pub fn as_ptr_mut(&mut self) -> *mut MNNScheduleConfig {
        self.inner
    }

    pub fn new() -> Self {
        unsafe {
            let inner = mnnsc_create();
            Self {
                inner,
                __marker: core::marker::PhantomData,
            }
        }
    }

    pub fn set_save_tensors(&mut self, save_tensors: &[&str]) -> Result<()> {
        let vec_cstring = save_tensors
            .iter()
            .map(|s| {
                std::ffi::CString::new(*s)
                    .map_err(|e| anyhow!("Failed to convert to cstr: {:?}", e))
            })
            .collect::<Result<Vec<_>>>()?;
        let vec_cstr = vec_cstring
            .iter()
            .map(|s| s.as_c_str().as_ptr())
            .collect::<Vec<_>>();
        unsafe { mnnsc_set_save_tensors(self.inner, vec_cstr.as_ptr(), vec_cstr.len()) }
        Ok(())
    }

    pub fn set_type(&mut self, forward_type: MNNForwardType) {
        unsafe {
            mnnsc_set_type(self.inner, forward_type);
        }
    }

    pub fn set_num_threads(&mut self, num_threads: i32) {
        unsafe {
            mnnsc_set_num_threads(self.inner, num_threads);
        }
    }

    pub fn set_mode(&mut self, mode: i32) {
        unsafe {
            mnnsc_set_mode(self.inner, mode);
        }
    }

    pub fn set_backup_type(&mut self, backup_type: MNNForwardType) {
        unsafe {
            mnnsc_set_backup_type(self.inner, backup_type);
        }
    }

    pub fn set_backend_config(&mut self, backend_config: &crate::BackendConfig) {
        unsafe {
            mnnsc_set_backend_config(self.inner, backend_config.as_ptr_mut());
        }
    }
}
