use crate::{prelude::*, AsTensorShape, Device, Ref, Tensor, TensorType};
use mnn_sys::HalideType;

#[derive(Debug, Copy, Clone)]
#[cfg_attr(windows, repr(i32))]
#[cfg_attr(unix, repr(u32))]
pub enum SessionMode {
    #[doc = "About CallBack, Default Session_Debug*/\n/** runSessionWithCallBack is allowed and can get internal op info"]
    Debug = mnn_sys::SessionMode::Session_Debug,
    #[doc = "runSessionWithCallBack is not valid and can't get any info of op in\nsession"]
    Release = mnn_sys::SessionMode::Session_Release,
    #[doc = "About input tenosr, Default Session_Input_Inside*/\n/** The input tensor is alloced by session, input data after session resized"]
    InputInside = mnn_sys::SessionMode::Session_Input_Inside,
    #[doc = "The input tensor is alloced by user, set input data before session\nresize"]
    InputUser = mnn_sys::SessionMode::Session_Input_User,
    #[doc = "The output tensor depends on session, and can't be separate used"]
    OutputInside = mnn_sys::SessionMode::Session_Output_Inside,
    #[doc = "The output tensor can be separated from session"]
    OutputUser = mnn_sys::SessionMode::Session_Output_User,
    #[doc = "Try Resize Session when create Session or not, default direct:"]
    ResizeDirect = mnn_sys::SessionMode::Session_Resize_Direct,
    #[doc = "Try Resize Session when create Session or not, default direct:"]
    ResizeDefer = mnn_sys::SessionMode::Session_Resize_Defer,
    #[doc = "Determine the Execution's forward type is determine by user or auto\ndetermine"]
    BackendFix = mnn_sys::SessionMode::Session_Backend_Fix,
    #[doc = "Determine the Execution's forward type is determine by user or auto\ndetermine"]
    BackendAuto = mnn_sys::SessionMode::Session_Backend_Auto,
    #[doc = "Determine static memory whether recyle in resizeSession or just cache the\nmemory"]
    MemoryCollect = mnn_sys::SessionMode::Session_Memory_Collect,
    #[doc = "Determine static memory whether recyle in resizeSession or just cache the\nmemory"]
    MemoryCache = mnn_sys::SessionMode::Session_Memory_Cache,
    #[doc = "Determine whether use codegen function"]
    CodegenDisable = mnn_sys::SessionMode::Session_Codegen_Disable,
    #[doc = "Determine whether use codegen function"]
    CodegenEnable = mnn_sys::SessionMode::Session_Codegen_Enable,
    #[doc = "Dynamic Reisze Optimization"]
    ResizeCheck = mnn_sys::SessionMode::Session_Resize_Check,
    #[doc = "Dynamic Reisze Optimization"]
    ResizeFix = mnn_sys::SessionMode::Session_Resize_Fix,
}

impl SessionMode {
    #[cfg(windows)]
    fn to_mnn_sys(&self) -> i32 {
        *self as i32
    }
    #[cfg(unix)]
    fn to_mnn_sys(&self) -> u32 {
        *self as u32
    }
}

#[repr(transparent)]
pub struct Interpreter {
    pub(crate) interpreter: *mut mnn_sys::Interpreter,
    pub(crate) __marker: PhantomData<()>,
}

unsafe impl Send for Interpreter {}

impl Interpreter {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        ensure!(path.exists(), ErrorKind::IOError);
        let path = path.to_str().ok_or_else(|| error!(ErrorKind::AsciiError))?;
        let c_path = std::ffi::CString::new(path).change_context(ErrorKind::AsciiError)?;
        let interpreter = unsafe { mnn_sys::Interpreter_createFromFile(c_path.as_ptr()) };
        ensure!(!interpreter.is_null(), ErrorKind::InterpreterError);
        Ok(Self {
            interpreter,
            __marker: PhantomData,
        })
    }

    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Result<Self> {
        let bytes = bytes.as_ref();
        let size = bytes.len();
        let interpreter =
            unsafe { mnn_sys::Interpreter_createFromBuffer(bytes.as_ptr().cast(), size) };
        ensure!(!interpreter.is_null(), ErrorKind::InterpreterError);
        Ok(Self {
            interpreter,
            __marker: PhantomData,
        })
    }

    pub fn set_session_mode(&mut self, mode: SessionMode) {
        unsafe { mnn_sys::Interpreter_setSessionMode(self.interpreter, mode.to_mnn_sys()) }
    }

    pub fn resize_session(&self, session: &mut crate::Session) {
        unsafe { mnn_sys::Interpreter_resizeSession(self.interpreter, session.session) }
    }

    pub fn resize_tensor<T: TensorType>(&self, tensor: &mut Tensor<T>, dims: impl AsTensorShape) {
        let dims = dims.as_tensor_shape();
        let dims_len = dims.size;
        unsafe {
            mnn_sys::Interpreter_resizeTensor(
                self.interpreter,
                tensor.tensor,
                dims.shape.as_ptr(),
                dims_len,
            )
        }
    }

    pub fn resize_tensor_by_nchw<T: TensorType>(
        &self,
        tensor: &mut Tensor<T>,
        batch: i32,
        channel: i32,
        height: i32,
        width: i32,
    ) {
        unsafe {
            mnn_sys::Interpreter_resizeTensorByNCHW(
                self.interpreter,
                tensor.tensor,
                batch,
                channel,
                height,
                width,
            )
        }
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

    pub fn input<'i, H: HalideType>(
        &'i self,
        session: &crate::Session,
        name: impl AsRef<str>,
    ) -> Result<Tensor<Ref<'i, Device<H>>>> {
        let name = name.as_ref();
        let c_name = std::ffi::CString::new(name).change_context(ErrorKind::AsciiError)?;
        let input = unsafe {
            mnn_sys::Interpreter_getSessionInput(
                self.interpreter,
                session.as_ptr_mut(),
                c_name.as_ptr(),
            )
        };
        ensure!(!input.is_null(), ErrorKind::TensorError; format!("Input tensor \"{name}\" not found"));
        let tensor = unsafe { Tensor::from_ptr(input) };
        ensure!(
            tensor.is_type_of::<H>(),
            ErrorKind::HalideTypeMismatch {
                got: std::any::type_name::<H>(),
            }
        );
        Ok(tensor)
    }

    pub fn output<'i, H: HalideType>(
        &'i self,
        session: &crate::Session,
        name: impl AsRef<str>,
    ) -> Result<Tensor<Ref<'_, Device<H>>>> {
        let name = name.as_ref();
        let c_name = std::ffi::CString::new(name).change_context(ErrorKind::AsciiError)?;
        let output = unsafe {
            mnn_sys::Interpreter_getSessionOutput(
                self.interpreter,
                session.as_ptr_mut(),
                c_name.as_ptr(),
            )
        };
        ensure!(!output.is_null(), ErrorKind::IOError);
        let tensor = unsafe { Tensor::from_ptr(output) };
        ensure!(
            tensor.is_type_of::<H>(),
            ErrorKind::HalideTypeMismatch {
                got: std::any::type_name::<H>(),
            }
        );
        Ok(tensor)
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

// impl core::fmt::Debug for TensorInfo<'_, '_> {
//     fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
//         f.debug_struct("TensorInfo")
//             .field("name", &self.name())
//             .field("tensor", &self.tensor().shape())
//             .finish()
//     }
// }

impl<'t, 'tl> TensorInfo<'t, 'tl> {
    pub fn name(&self) -> &'tl str {
        debug_assert!(!self.tensor_info.is_null());
        unsafe { (*self.tensor_info).name.to_cstr() }
            .to_str()
            .expect("Tensor name is not utf-8")
    }

    pub fn tensor<H: HalideType>(&self) -> Result<Tensor<Ref<'t, Device<H>>>> {
        debug_assert!(!self.tensor_info.is_null());
        unsafe { debug_assert!(!(*self.tensor_info).tensor.is_null()) };
        let tensor = unsafe { Tensor::from_ptr((*self.tensor_info).tensor.cast()) };
        ensure!(
            tensor.is_type_of::<H>(),
            ErrorKind::HalideTypeMismatch {
                got: std::any::type_name::<H>(),
            }
        );
        Ok(tensor)
    }
}

#[repr(transparent)]
pub struct TensorList<'t> {
    pub(crate) inner: *const mnn_sys::TensorInfoArray,
    pub(crate) __marker: PhantomData<&'t Interpreter>,
}

// impl<'t> core::fmt::Debug for TensorList<'t> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> core::fmt::Result {
//         f.debug_list().entries(self.iter()).finish()
//     }
// }

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

    // pub fn to_map(
    //     &'t self,
    // ) -> std::collections::HashMap<String, crate::Tensor<crate::Ref<'_, Device< H>>> {
    //     self.iter()
    //         .map(|t| (t.name().to_string(), t.tensor()))
    //         .collect()
    // }
    //

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
