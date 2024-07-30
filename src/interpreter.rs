use crate::prelude::*;
pub use mnn_sys::SessionMode;

#[repr(transparent)]
pub struct Interpreter {
    pub(crate) interpreter: *mut mnn_sys::Interpreter,
    pub(crate) __marker: PhantomData<()>,
}

impl Interpreter {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        ensure!(path.exists(), ErrorKind::IOError);
        let path = path.to_str().ok_or_else(|| error!(ErrorKind::AsciiError))?;
        let c_path = std::ffi::CString::new(path).change_context(ErrorKind::AsciiError)?;
        let interpreter = unsafe { mnn_sys::Interpreter_createFromFile(c_path.as_ptr()) };
        ensure!(!interpreter.is_null(), ErrorKind::IOError);
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
        schedule: &mut crate::ScheduleConfig,
    ) -> Result<crate::session::Session> {
        let session =
            unsafe { mnn_sys::Interpreter_createSession(self.interpreter, schedule.as_ptr_mut()) };
        Ok(unsafe { crate::session::Session::from_ptr(session) })
    }

    pub fn model_print_io(path: impl AsRef<std::path::Path>) -> Result<()> {
        let path = path.as_ref();
        crate::ensure!(path.exists(), ErrorKind::IOError);
        let path = path.to_str().ok_or_else(|| error!(ErrorKind::AsciiError))?;
        let c_path = std::ffi::CString::new(path).unwrap();
        unsafe { mnn_sys::modelPrintIO(c_path.as_ptr()) }
        Ok(())
    }

    pub fn get_inputs(&self, session: &crate::Session) -> TensorList {
        let inputs =
            unsafe { mnn_sys::Interpreter_getSessionInputAll(self.interpreter, session.session) };
        TensorList::from_raw(inputs)
    }

    pub fn get_input<'i>(
        &'i self,
        session: &crate::Session,
        name: impl AsRef<str>,
    ) -> Result<crate::TensorRef<'i, crate::Device>> {
        let name = name.as_ref();
        let c_name = std::ffi::CString::new(name).change_context(ErrorKind::AsciiError)?;
        let input = unsafe {
            mnn_sys::Interpreter_getSessionInput(
                self.interpreter,
                session.raw_mut(),
                c_name.as_ptr(),
            )
        };
        ensure!(!input.is_null(), ErrorKind::IOError);
        Ok(crate::TensorRef {
            tensor: input,
            __marker: PhantomData,
        })
    }

    pub fn get_output<'i>(
        &'i self,
        session: &crate::Session,
        name: impl AsRef<str>,
    ) -> Result<crate::TensorRef<'i, crate::Device>> {
        let name = name.as_ref();
        let c_name = std::ffi::CString::new(name).change_context(ErrorKind::AsciiError)?;
        let output = unsafe {
            mnn_sys::Interpreter_getSessionOutput(
                self.interpreter,
                session.raw_mut(),
                c_name.as_ptr(),
            )
        };
        ensure!(!output.is_null(), ErrorKind::IOError);
        // if output.is_null() {
        //     return Err(anyhow::anyhow!("Interpreter_getOutput failed")
        //         .context("Either output name is invalid or output is not found"));
        // }
        Ok(crate::TensorRef {
            tensor: output,
            __marker: PhantomData,
        })
    }

    pub fn run_session(&self, session: &crate::session::Session) -> Result<()> {
        let ret = unsafe { mnn_sys::Interpreter_runSession(self.interpreter, session.session) };
        ensure!(
            ret == mnn_sys::ErrorCode::ERROR_CODE_NO_ERROR,
            ErrorKind::InternalError(ret)
        );
        Ok(())
    }

    pub fn get_outputs(&self, session: &crate::session::Session) -> TensorList {
        let outputs =
            unsafe { mnn_sys::Interpreter_getSessionOutputAll(self.interpreter, session.session) };
        TensorList::from_raw(outputs)
    }
}

#[derive(Copy, Clone)]
#[repr(transparent)]
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
    pub fn name(&self) -> &'t str {
        debug_assert!(!self.tensor_info.is_null());
        unsafe { (*self.tensor_info).name.to_cstr() }
            .to_str()
            .expect("FIX ME later")
    }

    pub fn tensor(&self) -> crate::TensorRef<'t, crate::Device> {
        debug_assert!(!self.tensor_info.is_null());
        unsafe { debug_assert!(!(*self.tensor_info).tensor.is_null()) };
        let tensor = unsafe { (*self.tensor_info).tensor.cast() };
        crate::TensorRef {
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
