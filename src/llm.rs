use core::ffi::c_int;
use std::{ffi::CString, path::Path};

use crate::prelude::*;
use mnn_sys::llm;

#[derive(Debug)]
#[repr(transparent)]
pub struct Llm {
    inner: llm::LLM,
    __marker: PhantomData<()>,
}

impl Drop for Llm {
    fn drop(&mut self) {
        unsafe { llm::LLM_destroy(core::mem::transmute_copy(self)) };
    }
}

// impl Drop

impl Llm {
    /// Safety:
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let path = path.to_str().ok_or_else(|| ErrorKind::AsciiError)?;
        let path = CString::new(path).change_context(ErrorKind::AsciiError)?;
        let llm = unsafe { llm::LLM_create(path.as_ptr()) };
        Ok(Self {
            inner: llm,
            __marker: PhantomData,
        })
    }

    unsafe fn __memcpy<T>(&self) -> T {
        unsafe { core::mem::transmute_copy(self) }
    }

    pub fn chat(&mut self) -> Result<()> {
        unsafe { llm::LLM_chat(self.__memcpy()) };
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        unsafe { llm::LLM_reset(self.__memcpy()) };
        Ok(())
    }

    pub fn load(&mut self) -> Result<()> {
        unsafe { llm::LLM_load(self.__memcpy()) };
        Ok(())
    }

    pub fn generate(&mut self, ids: &[c_int]) -> Result<String> {
        let ret = unsafe { llm::LLM_generate(self.__memcpy(), ids.as_ptr(), ids.len()) };
        let ret: &[u8] = unsafe { core::slice::from_raw_parts(ret.data.cast(), ret.size) };
        Ok(String::from_utf8(ret.to_vec()).change_context(ErrorKind::AsciiError)?)
    }
}
