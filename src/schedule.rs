use mnn_sys::*;
use std::{ffi::CString, mem::ManuallyDrop};

use crate::{prelude::*, BackendConfig};

/// Backend used for running the model
///
/// The `ForwardType` enum is used to specify the backend that will be used for forward computation
/// in the MNN framework. Each variant corresponds to a different backend, which may be enabled
/// or disabled based on the features enabled in the build configuration.
///
/// # Variants
///
/// - `All`: Use all available backends.
/// - `Auto`: Automatically select the best backend based on the current environment and hardware.
/// - `CPU`: Use the CPU for computation.
/// - `Metal`: Use the Metal backend for computation (requires the `metal` feature).
/// - `OpenCL`: Use the OpenCL backend for computation (requires the `opencl` feature).
/// - `OpenGL`: Use the OpenGL backend for computation (requires the `opengl` feature).
/// - `Vulkan`: Use the Vulkan backend for computation (requires the `vulkan` feature).
/// - `CoreML`: Use the CoreML backend for computation (requires the `coreml` feature).
///
/// # Example
///
/// ```rust
/// use mnn::schedule::ForwardType;
///
/// let forward_type = ForwardType::Auto;
/// println!("Selected forward type: {:?}", forward_type);
/// ```
///
/// # Note
///
/// The availability of certain variants depends on the features enabled during the build.
/// For example, the `Metal` variant is only available if the `metal` feature is enabled.
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
    /// Convert the `ForwardType` enum to the corresponding C++ `MNNForwardType` enum.
    fn to_mnn_sys(self) -> MNNForwardType {
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

/// Configuration for scheduling the forward computation in MNN.
///
/// The `ScheduleConfig` struct is used to configure various parameters for scheduling the forward
/// computation in the MNN framework. It allows setting the type of backend, the number of threads,
/// the mode of computation, and other options.
///
/// # Example
///
/// ```rust
/// use mnn::schedule::{ScheduleConfig, ForwardType};
///
/// let mut config = ScheduleConfig::new();
/// config.set_type(ForwardType::Auto);
/// config.set_num_threads(4);
/// config.set_mode(0);
/// ```
///
/// # Fields
///
/// - `inner`: A raw pointer to the underlying `MNNScheduleConfig` structure.
/// - `backend_config`: Specifies backend-specific configurations.
/// - `__marker`: A marker to ensure the struct is `!Send` by default.
///
/// # Methods
///
/// - `new() -> Self`: Creates a new `ScheduleConfig` with default settings.
/// - `as_ptr_mut(&mut self) -> *mut MNNScheduleConfig`: Returns a mutable raw pointer to the underlying `MNNScheduleConfig`.
/// - `set_save_tensors(&mut self, save_tensors: &[&str]) -> Result<()>`: Sets the tensors to be saved during computation.
/// - `set_type(&mut self, forward_type: ForwardType)`: Sets the type of backend to be used for computation.
/// - `set_num_threads(&mut self, num_threads: i32)`: Sets the number of threads to be used for computation.
/// - `set_mode(&mut self, mode: i32)`: Sets the mode of computation.
/// - `set_backup_type(&mut self, backup_type: ForwardType)`: Sets the backup type of backend to be used if the primary backend fails.
/// - `set_backend_config(&mut self, backend_config: impl Into<Option<BackendConfig>>)`: Sets the backend-specific configuration.
///
/// # Safety
///
/// The `ScheduleConfig` struct contains raw pointers and interacts with the underlying C API of MNN.
/// Users should be cautious when using this struct to avoid undefined behavior.
///
/// # Warning
///
/// **Warning:** The `Drop` implementation for `ScheduleConfig` ensures that the underlying `MNNScheduleConfig`
/// is properly destroyed when the struct goes out of scope. Users should not manually free the `inner` pointer.
#[derive(Debug)]
pub struct ScheduleConfig {
    pub(crate) inner: *mut MNNScheduleConfig,
    pub(crate) backend_config: Option<BackendConfig>,
    pub(crate) __marker: core::marker::PhantomData<()>,
}

impl Clone for ScheduleConfig {
    fn clone(&self) -> Self {
        unsafe {
            let inner = mnnsc_clone(self.inner);
            Self {
                inner,
                backend_config: self.backend_config.clone(),
                __marker: core::marker::PhantomData,
            }
        }
    }
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
    /// Returns a mutable raw pointer to the underlying `MNNScheduleConfig`.
    pub fn as_ptr_mut(&mut self) -> *mut MNNScheduleConfig {
        self.inner
    }

    /// Creates a new `ScheduleConfig` with default settings.
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

    /// Sets the tensors to be saved during computation.
    ///
    /// # Arguments
    ///
    /// - `save_tensors`: A slice of tensor names to be saved.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the tensor names contain null bytes.
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

    /// Sets the type of backend to be used for computation.
    ///
    /// # Arguments
    ///
    /// - `forward_type`: The type of backend to be used.
    pub fn set_type(&mut self, forward_type: ForwardType) {
        unsafe {
            mnnsc_set_type(self.inner, forward_type.to_mnn_sys());
        }
    }

    /// Sets the number of threads to be used for computation.
    ///
    /// # Arguments
    ///
    /// - `num_threads`: The number of threads to be used.
    pub fn set_num_threads(&mut self, num_threads: i32) {
        unsafe {
            mnnsc_set_num_threads(self.inner, num_threads);
        }
    }

    /// Sets the mode of computation.
    ///
    /// # Arguments
    ///
    /// - `mode`: The mode of computation.
    pub fn set_mode(&mut self, mode: i32) {
        unsafe {
            mnnsc_set_mode(self.inner, mode);
        }
    }

    /// Sets the backup type of backend to be used if the primary backend fails.
    ///
    /// # Arguments
    ///
    /// - `backup_type`: The backup type of backend to be used.
    pub fn set_backup_type(&mut self, backup_type: ForwardType) {
        unsafe {
            mnnsc_set_backup_type(self.inner, backup_type.to_mnn_sys());
        }
    }

    /// Sets the backend-specific configuration.
    ///
    /// # Arguments
    ///
    /// - `backend_config`: specifies additional backend-specific configurations.
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

#[derive(Debug)]
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
