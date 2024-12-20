//! The backend module contains the data types for the backend configuration

use crate::prelude::*;
use std::str::FromStr;

use mnn_sys::*;

/// BackendConfig is a struct that holds the configuration for the backend
/// memory: [MemoryMode]
/// power: [PowerMode]
/// precision: [PrecisionMode]
#[repr(transparent)]
pub struct BackendConfig {
    pub(crate) inner: *mut MNNBackendConfig,
    __marker: core::marker::PhantomData<()>,
}

impl core::fmt::Debug for BackendConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendConfig")
            .field("memory", &self.get_memory_mode())
            .field("power", &self.get_power_mode())
            .field("precision", &self.get_precision_mode())
            .finish()
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for BackendConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("BackendConfig", 3)?;
        state.serialize_field("memory", &self.get_memory_mode())?;
        state.serialize_field("power", &self.get_power_mode())?;
        state.serialize_field("precision", &self.get_precision_mode())?;
        state.end()
    }
}

impl Clone for BackendConfig {
    fn clone(&self) -> Self {
        unsafe {
            let inner = mnn_sys::mnnbc_clone(self.inner);
            Self {
                inner,
                __marker: core::marker::PhantomData,
            }
        }
    }
}

impl Drop for BackendConfig {
    fn drop(&mut self) {
        unsafe {
            mnn_sys::mnnbc_destroy(self.inner);
        }
    }
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// PowerModes depend on if the specific backend has support for it
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PowerMode {
    /// Low power mode
    Low,
    /// Normal power mode
    Normal,
    /// High power mode
    High,
}

impl PowerMode {
    fn to_mnn_sys(self) -> mnn_sys::PowerMode {
        match self {
            Self::Low => mnn_sys::PowerMode::Power_Low,
            Self::Normal => mnn_sys::PowerMode::Power_Normal,
            Self::High => mnn_sys::PowerMode::Power_High,
        }
    }

    /// Returns a string representation of the power mode
    pub fn to_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Normal => "normal",
            Self::High => "high",
        }
    }

    fn from_mnn_sys(mode: mnn_sys::PowerMode) -> Self {
        match mode {
            mnn_sys::PowerMode::Power_Low => Self::Low,
            mnn_sys::PowerMode::Power_Normal => Self::Normal,
            mnn_sys::PowerMode::Power_High => Self::High,
            _ => Self::Normal,
        }
    }
}

impl FromStr for PowerMode {
    type Err = MNNError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "low" => Ok(Self::Low),
            "normal" => Ok(Self::Normal),
            "high" => Ok(Self::High),
            _ => {
                Err(error!(ErrorKind::ParseError)
                    .attach_printable(format!("invalid power mode: {s}")))
            }
        }
    }
}

impl FromStr for MemoryMode {
    type Err = MNNError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "low" => Ok(Self::Low),
            "normal" => Ok(Self::Normal),
            "high" => Ok(Self::High),
            _ => {
                Err(error!(ErrorKind::ParseError)
                    .attach_printable(format!("invalid memory mode: {s}")))
            }
        }
    }
}

impl FromStr for PrecisionMode {
    type Err = MNNError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "low" => Ok(Self::Low),
            "normal" => Ok(Self::Normal),
            "high" => Ok(Self::High),
            "low_bf16" => Ok(Self::LowBf16),
            _ => Err(error!(ErrorKind::ParseError)
                .attach_printable(format!("invalid precision mode: {s}"))),
        }
    }
}

/// MemoryModes depend on if the specific backend has support for it
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MemoryMode {
    /// Low memory mode
    Low,
    /// Normal memory mode
    Normal,
    /// High memory mode
    High,
}

impl MemoryMode {
    fn to_mnn_sys(self) -> mnn_sys::MemoryMode {
        match self {
            Self::Low => mnn_sys::MemoryMode::Memory_Low,
            Self::Normal => mnn_sys::MemoryMode::Memory_Normal,
            Self::High => mnn_sys::MemoryMode::Memory_High,
        }
    }

    /// Returns a string representation of the memory mode
    pub fn to_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Normal => "normal",
            Self::High => "high",
        }
    }

    fn from_mnn_sys(mode: mnn_sys::MemoryMode) -> Self {
        match mode {
            mnn_sys::MemoryMode::Memory_Low => Self::Low,
            mnn_sys::MemoryMode::Memory_Normal => Self::Normal,
            mnn_sys::MemoryMode::Memory_High => Self::High,
            _ => Self::Normal,
        }
    }
}

/// PrecisionModes depend on if the specific backend has support for it
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PrecisionMode {
    /// Normal precision mode
    Normal = 0,
    /// High precision mode
    High,
    /// Low precision mode
    Low,
    /// Low precision mode with BF16
    LowBf16,
}
impl PrecisionMode {
    pub(crate) fn to_mnn_sys(self) -> mnn_sys::PrecisionMode {
        match self {
            Self::LowBf16 => mnn_sys::PrecisionMode::Precision_Low_BF16,
            Self::Low => mnn_sys::PrecisionMode::Precision_Low,
            Self::Normal => mnn_sys::PrecisionMode::Precision_Normal,
            Self::High => mnn_sys::PrecisionMode::Precision_High,
        }
    }

    /// Returns a string representation of the precision mode
    pub fn to_str(self) -> &'static str {
        match self {
            Self::LowBf16 => "low_bf16",
            Self::Low => "low",
            Self::Normal => "normal",
            Self::High => "high",
        }
    }

    fn from_mnn_sys(mode: mnn_sys::PrecisionMode) -> Self {
        match mode {
            mnn_sys::PrecisionMode::Precision_Low_BF16 => Self::LowBf16,
            mnn_sys::PrecisionMode::Precision_Low => Self::Low,
            mnn_sys::PrecisionMode::Precision_Normal => Self::Normal,
            mnn_sys::PrecisionMode::Precision_High => Self::High,
            _ => Self::Normal,
        }
    }
}

impl BackendConfig {
    /// Create a new backend config
    pub fn new() -> Self {
        unsafe {
            let inner = mnnbc_create();
            Self {
                inner,
                __marker: core::marker::PhantomData,
            }
        }
    }

    /// Sets the [MemoryMode] for the backend
    pub fn set_memory_mode(&mut self, mode: MemoryMode) {
        unsafe {
            mnn_sys::mnnbc_set_memory_mode(self.inner, mode.to_mnn_sys());
        }
    }

    /// Sets the [MemoryMode] for the backend
    pub fn with_memory_mode(mut self, mode: MemoryMode) -> Self {
        self.set_memory_mode(mode);
        self
    }

    /// Gets the [MemoryMode] for the backend
    pub fn get_memory_mode(&self) -> MemoryMode {
        unsafe { MemoryMode::from_mnn_sys(mnn_sys::mnnbc_get_memory_mode(self.inner)) }
    }

    /// Sets the [PowerMode] for the backend
    pub fn set_power_mode(&mut self, mode: PowerMode) {
        unsafe {
            mnn_sys::mnnbc_set_power_mode(self.inner, mode.to_mnn_sys());
        }
    }

    /// Sets the [PowerMode] for the backend
    pub fn with_power_mode(mut self, mode: PowerMode) -> Self {
        self.set_power_mode(mode);
        self
    }

    /// Gets the [PowerMode] for the backend
    pub fn get_power_mode(&self) -> PowerMode {
        unsafe { PowerMode::from_mnn_sys(mnn_sys::mnnbc_get_power_mode(self.inner)) }
    }

    /// Sets the [PrecisionMode] for the backend
    pub fn set_precision_mode(&mut self, mode: PrecisionMode) {
        unsafe {
            mnn_sys::mnnbc_set_precision_mode(self.inner, mode.to_mnn_sys());
        }
    }

    /// Sets the [PrecisionMode] for the backend
    pub fn with_precision_mode(mut self, mode: PrecisionMode) -> Self {
        self.set_precision_mode(mode);
        self
    }

    /// Gets the [PrecisionMode] for the backend
    pub fn get_precision_mode(&self) -> PrecisionMode {
        unsafe { PrecisionMode::from_mnn_sys(mnn_sys::mnnbc_get_precision_mode(self.inner)) }
    }

    /// Sets the flags for the backend
    /// What the flag represents is depends on each backend or isn't documented
    pub fn set_flags(&mut self, flags: usize) {
        unsafe {
            mnn_sys::mnnbc_set_flags(self.inner, flags);
        }
    }

    /// Sets the flags for the backend
    pub fn with_flags(mut self, flags: usize) -> Self {
        self.set_flags(flags);
        self
    }

    /// # Safety
    /// This just binds to the underlying unsafe api and should be used only if you know what you
    /// are doing
    pub unsafe fn set_shared_context(&mut self, shared_context: *mut libc::c_void) {
        unsafe {
            mnn_sys::mnnbc_set_shared_context(self.inner, shared_context);
        }
    }

    /// # Safety
    /// This just binds to the underlying unsafe api and should be used only if you know what you
    /// are doing
    pub unsafe fn with_shared_context(mut self, shared_context: *mut libc::c_void) -> Self {
        self.set_shared_context(shared_context);
        self
    }
}

#[test]
fn test_backend_config() {
    let mut config = BackendConfig::new();
    config.set_memory_mode(MemoryMode::Low);
    config.set_power_mode(PowerMode::Low);
    config.set_precision_mode(PrecisionMode::Low);
    let config = std::hint::black_box(config.clone());
    assert_eq!(config.get_memory_mode(), MemoryMode::Low);
    assert_eq!(config.get_power_mode(), PowerMode::Low);
    assert_eq!(config.get_precision_mode(), PrecisionMode::Low);
    let config = config
        .with_memory_mode(MemoryMode::Normal)
        .with_power_mode(PowerMode::Normal)
        .with_precision_mode(PrecisionMode::Normal);
    assert_eq!(config.get_memory_mode(), MemoryMode::Normal);
    assert_eq!(config.get_power_mode(), PowerMode::Normal);
    assert_eq!(config.get_precision_mode(), PrecisionMode::Normal);
}
