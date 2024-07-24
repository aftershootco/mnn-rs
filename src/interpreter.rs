use anyhow::Result;
use core::pin::Pin;
use mnn_sys::MNN::Interpreter_SessionMode;
use std::{ffi::CString, path::Path};
pub struct Interpreter {
    pub(crate) inner: *mut mnn_sys::MNN::Interpreter,
}

impl Interpreter {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let c_path = CString::new(
            path.to_str()
                .ok_or_else(|| anyhow::anyhow!("Failed to convert path to c_str"))?,
        )?;
        let inner = unsafe { mnn_sys::MNN::Interpreter::createFromFile(c_path.as_ptr()) };
        Ok(Self { inner })
    }
    pub fn set_session_mode(&mut self, mode: Interpreter_SessionMode) -> &mut Self {
        let this = Pin::new(unsafe { &mut *self.inner });
        mnn_sys::MNN::Interpreter::setSessionMode(this, mode);
        self
    }

    pub fn create_session(&mut self) -> Result<()> {
        let this = Pin::new(unsafe { &mut *self.inner });
        let config = mnn_sys::MNN::glueScheduleConfigCreate();
        let config = unsafe { &*config };
        let status = mnn_sys::MNN::Interpreter::createSession(this, config);
        if status.is_null() {
            Err(anyhow::anyhow!("Failed to create session"))
        } else {
            Ok(())
        }
    }
}
