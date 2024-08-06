use crate::{BackendConfig, ScheduleConfig};

// use core::ffi::c_int;
// use std::{ffi::CString, path::Path};
//
// use crate::prelude::*;
// use mnn_sys::llm;
//
// #[derive(Debug)]
// #[repr(transparent)]
// pub struct Llm {
//     inner: llm::LLM,
//     __marker: PhantomData<()>,
// }
//
// impl Drop for Llm {
//     fn drop(&mut self) {
//         unsafe { llm::LLM_destroy(core::mem::transmute_copy(self)) };
//     }
// }
//
// // impl Drop
//
// impl Llm {
//     /// Safety:
//     pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
//         let path = path.as_ref();
//         let path = path.to_str().ok_or_else(|| ErrorKind::AsciiError)?;
//         let path = CString::new(path).change_context(ErrorKind::AsciiError)?;
//         let llm = unsafe { llm::LLM_create(path.as_ptr()) };
//         ensure!(!llm.llm.is_null(), ErrorKind::LlmError);
//         Ok(Self {
//             inner: llm,
//             __marker: PhantomData,
//         })
//     }
//
//     unsafe fn __memcpy<T>(&self) -> T {
//         unsafe { core::mem::transmute_copy(self) }
//     }
//
//     pub fn chat(&mut self) -> Result<()> {
//         unsafe { llm::LLM_chat(self.__memcpy()) };
//         Ok(())
//     }
//
//     pub fn reset(&mut self) -> Result<()> {
//         unsafe { llm::LLM_reset(self.__memcpy()) };
//         Ok(())
//     }
//
//     pub fn load(&mut self) -> Result<()> {
//         unsafe { llm::LLM_load(self.__memcpy()) };
//         Ok(())
//     }
//
//     pub fn generate(&mut self, ids: &[c_int], end_with: impl AsRef<str>) -> Result<String> {
//         let end_with = CString::new(end_with.as_ref()).change_context(ErrorKind::AsciiError)?;
//         let out = unsafe { llm::create_ostream(2) };
//         let mut ret = unsafe {
//             llm::LLM_generate(
//                 self.__memcpy(),
//                 ids.as_ptr(),
//                 ids.len(),
//                 out,
//                 end_with.as_ptr(),
//             )
//         };
//         let ret_ptr = unsafe { llm::LLMString_as_str(core::ptr::addr_of_mut!(ret)) };
//         let ret = unsafe { core::ffi::CStr::from_ptr(ret_ptr).to_string_lossy() };
//         // Ok(String::from_utf8(ret.to_vec()).change_context(ErrorKind::AsciiError)?)
//         Ok(ret.into_owned())
//     }
//
//     // pub fn response(&mut self,
// }
pub struct LLM {}
impl LLM {
    pub fn init() {
        let mut config = ScheduleConfig::new();
        let mut backend_config = BackendConfig::new();
        config.set_type(mnn_sys::MNNForwardType::MNN_FORWARD_OPENCL);
        config.set_backend_config(&backend_config);
    }
}
