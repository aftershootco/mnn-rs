use mnn_sys::*;
use std::ffi::CString;

use crate::prelude::*;
pub struct ScheduleConfig {
    pub(crate) inner: *mut MNNScheduleConfig,
    pub(crate) __marker: core::marker::PhantomData<()>,
}

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
    #[cfg(feature = "opengl")]
    OpenGL,
    #[cfg(feature = "vulkan")]
    Vulkan,
    #[cfg(feature = "coreml")]
    CoreML,
}

impl ForwardType {
    fn to_mnn_sys(&self) -> MNNForwardType {
        match self {
            ForwardType::Auto => MNNForwardType::MNN_FORWARD_AUTO,
            ForwardType::All => MNNForwardType::MNN_FORWARD_ALL,
            ForwardType::CPU => MNNForwardType::MNN_FORWARD_CPU,
            #[cfg(feature = "metal")]
            ForwardType::Metal => MNNForwardType::MNN_FORWARD_METAL,
            #[cfg(feature = "opencl")]
            ForwardType::OpenCL => MNNForwardType::MNN_FORWARD_OPENCL,
            #[cfg(feature = "opengl")]
            ForwardType::OpenGL => MNNForwardType::MNN_FORWARD_OPENGL,
            #[cfg(feature = "vulkan")]
            ForwardType::Vulkan => MNNForwardType::MNN_FORWARD_VULKAN,
            #[cfg(feature = "coreml")]
            ForwardType::CoreML => MNNForwardType::MNN_FORWARD_NN,
        }
    }
    #[cfg(feature = "parse")]
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

#[cfg(feature = "parse")]
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

    pub fn set_backend_config(&mut self, backend_config: &crate::BackendConfig) {
        unsafe {
            mnnsc_set_backend_config(self.inner, backend_config.as_ptr_mut());
        }
    }
}
