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

    pub fn inputs(&self, session: &crate::Session) -> TensorList {
        let inputs =
            unsafe { mnn_sys::Interpreter_getSessionInputAll(self.interpreter, session.session) };
        TensorList::from_ptr(inputs)
    }

    pub fn input<'i>(
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

    pub fn output<'i>(
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

    pub fn outputs(&self, session: &crate::session::Session) -> TensorList {
        let outputs =
            unsafe { mnn_sys::Interpreter_getSessionOutputAll(self.interpreter, session.session) };
        TensorList::from_ptr(outputs)
    }
}

#[repr(transparent)]
pub struct TensorInfo<'t, 'tl> {
    pub(crate) tensor_info: *mut mnn_sys::TensorInfo,
    pub(crate) __marker: PhantomData<&'tl TensorList<'t>>,
}

impl core::fmt::Debug for TensorInfo<'_, '_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TensorInfo")
            .field("name", &self.name())
            .field("tensor", &self.tensor().shape())
            .finish()
    }
}

impl<'t, 'tl> TensorInfo<'t, 'tl> {
    pub fn name(&self) -> &'tl str {
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

#[repr(transparent)]
pub struct TensorList<'t> {
    pub(crate) inner: *const mnn_sys::TensorInfoArray,
    pub(crate) __marker: PhantomData<&'t Interpreter>,
}

impl<'t> core::fmt::Debug for TensorList<'t> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl Drop for TensorList<'_> {
    fn drop(&mut self) {
        unsafe { mnn_sys::destroyTensorInfoArray(self.inner.cast_mut()) }
    }
}

impl<'t> TensorList<'t> {
    pub fn from_ptr(inner: *const mnn_sys::TensorInfoArray) -> Self {
        Self {
            inner,
            __marker: PhantomData,
        }
    }

    pub fn to_map(
        &'t self,
    ) -> std::collections::HashMap<String, crate::TensorRef<'t, crate::Device>> {
        self.iter()
            .map(|t| (t.name().to_string(), t.tensor()))
            .collect()
    }

    pub fn size(&self) -> usize {
        unsafe { (*self.inner).size }
    }

    pub fn get(&self, index: usize) -> Option<TensorInfo<'t, '_>> {
        if index >= self.size() {
            return None;
        } else {
            let gtinfo = unsafe { mnn_sys::getTensorInfoArray(self.inner, index) };
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

pub struct TensorListIter<'t, 'tl> {
    tensor_list: &'tl TensorList<'t>,
    idx: usize,
}
impl<'t, 'tl> Iterator for TensorListIter<'t, 'tl> {
    type Item = TensorInfo<'t, 'tl>;
    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.idx;
        self.idx += 1;
        self.tensor_list.get(idx)
    }
}
