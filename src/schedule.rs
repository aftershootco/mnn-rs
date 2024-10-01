use mnn_sys::*;
use std::{ffi::CString, mem::ManuallyDrop};

use crate::{prelude::*, BackendConfig};

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub enum ForwardType {
    All,
    #[default]
    Auto,
    CPU,
    #[cfg(feature = "metal")]
    Metal,
    #[cfg(feature = "opencl")]
    OpenCL,
    #[cfg(feature = "onednn")]
    OneDNN,
    #[cfg(feature = "opengl")]
    OpenGL,
    #[cfg(feature = "vulkan")]
    Vulkan,
    #[cfg(feature = "coreml")]
    CoreML,
}

impl ForwardType {
    fn to_mnn_sys(self) -> MNNForwardType {
        match self {
            ForwardType::Auto => MNNForwardType::MNN_FORWARD_AUTO,
            ForwardType::All => MNNForwardType::MNN_FORWARD_ALL,
            ForwardType::CPU => MNNForwardType::MNN_FORWARD_CPU,
            #[cfg(feature = "metal")]
            ForwardType::Metal => MNNForwardType::MNN_FORWARD_METAL,
            #[cfg(feature = "opencl")]
            ForwardType::OpenCL => MNNForwardType::MNN_FORWARD_OPENCL,
            #[cfg(feature = "onednn")]
            ForwardType::OneDNN => MNNForwardType::MNN_FORWARD_ONEDNN,
            #[cfg(feature = "opengl")]
            ForwardType::OpenGL => MNNForwardType::MNN_FORWARD_OPENGL,
            #[cfg(feature = "vulkan")]
            ForwardType::Vulkan => MNNForwardType::MNN_FORWARD_VULKAN,
            #[cfg(feature = "coreml")]
            ForwardType::CoreML => MNNForwardType::MNN_FORWARD_NN,
        }
    }

    fn list() -> Vec<&'static str> {
        vec![
            "auto",
            "all",
            "cpu",
            #[cfg(feature = "metal")]
            "metal",
            #[cfg(feature = "opencl")]
            "opencl",
            #[cfg(feature = "opengl")]
            "opengl",
            #[cfg(feature = "vulkan")]
            "vulkan",
            #[cfg(feature = "coreml")]
            "coreml",
        ]
    }
}

impl core::str::FromStr for ForwardType {
    type Err = MNNError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(ForwardType::Auto),
            "all" => Ok(ForwardType::All),
            "cpu" => Ok(ForwardType::CPU),
            #[cfg(feature = "metal")]
            "metal" => Ok(ForwardType::Metal),
            #[cfg(feature = "opencl")]
            "opencl" => Ok(ForwardType::OpenCL),
            #[cfg(feature = "opengl")]
            "opengl" => Ok(ForwardType::OpenGL),
            #[cfg(feature = "vulkan")]
            "vulkan" => Ok(ForwardType::Vulkan),
            #[cfg(feature = "coreml")]
            "coreml" => Ok(ForwardType::CoreML),
            _ => Err(MNNError::new(crate::ErrorKind::ParseError)
                .attach_printable(format!(
                    "Invalid ForwardType: {s}, maybe you might need to enable feature {s}"
                ))
                .attach_printable(format!(
                    "Valid ForwardType: {}",
                    ForwardType::list().join(", ")
                ))),
        }
    }
}

// #[derive(Debug)]
pub struct ScheduleConfig {
    pub(crate) inner: *mut MNNScheduleConfig,
    pub(crate) backend_config: Option<BackendConfig>,
    pub(crate) __marker: core::marker::PhantomData<()>,
}

impl Drop for ScheduleConfig {
    fn drop(&mut self) {
        unsafe {
            mnn_sys::mnnsc_destroy(self.inner);
        }
    }
}

unsafe impl Send for ScheduleConfig {}

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self::new()
    }
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
                backend_config: None,
                __marker: core::marker::PhantomData,
            }
        }
    }

    pub fn set_save_tensors(&mut self, save_tensors: &[&str]) -> Result<()> {
        let vec_cstring = save_tensors
            .iter()
            .map(|s| std::ffi::CString::new(*s).map_err(|e| error!(ErrorKind::AsciiError, e)))
            .collect::<Result<Vec<_>>>()?;
        let vec_cstr = vec_cstring
            .iter()
            .map(|s: &CString| s.as_c_str().as_ptr())
            .collect::<Vec<_>>();
        unsafe { mnnsc_set_save_tensors(self.inner, vec_cstr.as_ptr(), vec_cstr.len()) }
        Ok(())
    }

    pub fn set_type(&mut self, forward_type: ForwardType) {
        unsafe {
            mnnsc_set_type(self.inner, forward_type.to_mnn_sys());
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

    pub fn set_backup_type(&mut self, backup_type: ForwardType) {
        unsafe {
            mnnsc_set_backup_type(self.inner, backup_type.to_mnn_sys());
        }
    }

    pub fn set_backend_config(&mut self, backend_config: impl Into<Option<BackendConfig>>) {
        self.backend_config = backend_config.into();
        let ptr = if let Some(ref b) = self.backend_config {
            b.inner
        } else {
            core::ptr::null_mut()
        };
        unsafe {
            mnnsc_set_backend_config(self.inner, ptr);
        }
    }
}

pub struct ScheduleConfigs {
    pub(crate) inner: Vec<*const MNNScheduleConfig>,
    pub(crate) backend_configs: Vec<Option<BackendConfig>>,
}

impl Drop for ScheduleConfigs {
    fn drop(&mut self) {
        unsafe {
            for i in self.inner.iter() {
                mnnsc_destroy(*i.cast());
            }
        }
    }
}

impl ScheduleConfigs {
    pub fn push(&mut self, config: ScheduleConfig) {
        let mut config = ManuallyDrop::new(config);
        self.inner.push(config.inner);
        self.backend_configs.push(config.backend_config.take());
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Vec::with_capacity(capacity),
            backend_configs: Vec::with_capacity(capacity),
        }
    }

    pub const fn new() -> Self {
        Self {
            inner: Vec::new(),
            backend_configs: Vec::new(),
        }
    }
}

impl Default for ScheduleConfigs {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<ScheduleConfig> for ScheduleConfigs {
    fn from_iter<T: IntoIterator<Item = ScheduleConfig>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let mut ret = Self::with_capacity(iter.size_hint().1.unwrap_or_default());
        iter.for_each(|item| {
            ret.push(item);
        });
        ret
    }
}

unsafe impl Send for ScheduleConfigs {}
