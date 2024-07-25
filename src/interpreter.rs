use anyhow::Result;
use core::marker::PhantomData;
pub use mnn_sys::SessionMode;
pub struct Interpreter {
    pub(crate) interpreter: *mut mnn_sys::Interpreter,
    pub(crate) __marker: PhantomData<()>,
}

impl Interpreter {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            anyhow::bail!("File not found: {:?}", path);
        }
        let path = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Failed to convert to cstr"))?;
        let c_path = std::ffi::CString::new(path)?;
        let interpreter = unsafe { mnn_sys::Interpreter_createFromFile(c_path.as_ptr()) };
        if interpreter.is_null() {
            anyhow::bail!("Interpreter_createFromFile failed");
        }
        Ok(Self {
            interpreter,
            __marker: PhantomData,
        })
    }

    pub fn set_session_mode(&mut self, mode: mnn_sys::SessionMode) {
        unsafe { mnn_sys::Interpreter_setSessionMode(self.interpreter, mode) }
    }

    pub fn create_session(
        &mut self,
        schedule: &mnn_sys::ScheduleConfig,
    ) -> Result<crate::session::Session> {
        let session = unsafe { mnn_sys::Interpreter_createSession(self.interpreter, schedule) };
        Ok(unsafe { crate::session::Session::from_ptr(session) })
    }
    pub fn model_print_io(path: impl AsRef<std::path::Path>) -> Result<()> {
        let path = path.as_ref();
        if !path.exists() {
            anyhow::bail!("File not found: {:?}", path);
        }
        let path = path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Failed to convert to cstr"))
            .unwrap();
        let c_path = std::ffi::CString::new(path).unwrap();
        unsafe { mnn_sys::modelPrintIO(c_path.as_ptr()) }
        Ok(())
    }

    pub fn get_inputs(&self, session: &crate::Session) -> TensorList {
        let inputs =
            unsafe { mnn_sys::Interpreter_getSessionInputAll(self.interpreter, session.session) };
        TensorList::from_raw(inputs)
    }

    pub fn run_session(&self, session: &crate::session::Session) -> Result<()> {
        let ret = unsafe { mnn_sys::Interpreter_runSession(self.interpreter, session.session) };
        if ret != mnn_sys::ErrorCode::ERROR_CODE_NO_ERROR {
            anyhow::bail!("Interpreter_runSession failed");
        }
        Ok(())
    }

    pub fn get_outputs(&self, session: &crate::session::Session) -> TensorList {
        let outputs =
            unsafe { mnn_sys::Interpreter_getSessionOutputAll(self.interpreter, session.session) };
        TensorList::from_raw(outputs)
    }
}

#[derive(Copy, Clone)]
pub struct TensorInfo<'t> {
    pub(crate) tensor_info: *mut mnn_sys::TensorInfo,
    pub(crate) __marker: PhantomData<&'t TensorList>,
}

impl core::fmt::Debug for TensorInfo<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TensorInfo")
            .field("name", &self.name())
            .field("tensor", &self.tensor().shape())
            .finish()
    }
}

impl<'t> TensorInfo<'t> {
    pub fn name(&'t self) -> &'t str {
        debug_assert!(!self.tensor_info.is_null());
        unsafe { (*self.tensor_info).name.to_cstr() }
            .to_str()
            .expect("FIX ME")
    }

    pub fn tensor(&'t self) -> crate::Tensor<crate::Device> {
        debug_assert!(!self.tensor_info.is_null());
        unsafe { debug_assert!(!(*self.tensor_info).tensor.is_null()) };
        let tensor = unsafe { (*self.tensor_info).tensor.cast() };
        crate::Tensor {
            tensor,
            __marker: PhantomData,
        }
    }
}

pub struct TensorList {
    pub(crate) inner: mnn_sys::TensorInfoArray,
    pub(crate) __marker: PhantomData<()>,
}

impl core::fmt::Debug for TensorList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl Drop for TensorList {
    fn drop(&mut self) {
        unsafe { mnn_sys::destroyTensorInfoArray(&mut self.inner) }
    }
}

impl TensorList {
    pub fn from_raw(inner: mnn_sys::TensorInfoArray) -> Self {
        Self {
            inner,
            __marker: PhantomData,
        }
    }

    pub fn size(&self) -> usize {
        self.inner.size
    }

    pub fn get<'t>(&'t self, index: usize) -> Option<TensorInfo<'t>> {
        if index >= self.size() {
            return None;
        } else {
            let gtinfo =
                unsafe { mnn_sys::getTensorInfoArray(core::ptr::addr_of!(self.inner), index) };
            if !gtinfo.is_null() {
                Some(TensorInfo {
                    tensor_info: gtinfo,
                    __marker: PhantomData,
                })
            } else {
                None
            }
        }
    }

    pub fn iter(&self) -> TensorListIter {
        TensorListIter {
            tensor_list: self,
            idx: 0,
        }
    }
}

pub struct TensorListIter<'t> {
    tensor_list: &'t TensorList,
    idx: usize,
}
impl<'t> Iterator for TensorListIter<'t> {
    type Item = TensorInfo<'t>;
    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.idx;
        self.idx += 1;
        self.tensor_list.get(idx)
    }
}
