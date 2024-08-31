use crate::prelude::*;
use std::str::FromStr;

use mnn_sys::*;

#[repr(transparent)]
pub struct BackendConfig {
    pub inner: *mut MNNBackendConfig,
    __marker: core::marker::PhantomData<()>,
}

#[derive(Debug, Clone, Copy)]
pub enum PowerMode {
    Low,
    Normal,
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

#[derive(Debug, Clone, Copy)]
pub enum MemoryMode {
    Low,
    Normal,
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
}

#[derive(Debug, Clone, Copy)]
pub enum PrecisionMode {
    Normal = 0,
    High,
    Low,
    LowBf16,
}
impl PrecisionMode {
    fn to_mnn_sys(self) -> mnn_sys::PrecisionMode {
        match self {
            Self::LowBf16 => mnn_sys::PrecisionMode::Precision_Low_BF16,
            Self::Low => mnn_sys::PrecisionMode::Precision_Low,
            Self::Normal => mnn_sys::PrecisionMode::Precision_Normal,
            Self::High => mnn_sys::PrecisionMode::Precision_High,
        }
    }
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
            mnn_sys::mnnbc_set_memory_mode(self.inner, mode.to_mnn_sys());
        }
    }

    pub fn set_power_mode(&mut self, mode: PowerMode) {
        unsafe {
            mnn_sys::mnnbc_set_power_mode(self.inner, mode.to_mnn_sys());
        }
    }

    pub fn set_precision_mode(&mut self, mode: PrecisionMode) {
        unsafe {
            mnn_sys::mnnbc_set_precision_mode(self.inner, mode.to_mnn_sys());
        }
    }

    pub fn set_flags(&mut self, flags: usize) {
        unsafe {
            mnn_sys::mnnbc_set_flags(self.inner, flags);
        }
    }

    pub unsafe fn set_shared_context(&mut self, shared_context: *mut libc::c_void) {
        unsafe {
            mnn_sys::mnnbc_set_shared_context(self.inner, shared_context);
        }
    }
}
