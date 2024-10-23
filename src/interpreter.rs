use std::{ffi::CStr, path::Path, sync::Arc};

use crate::{
    prelude::*, AsTensorShape, Device, RawTensor, Ref, RefMut, ScheduleConfig, Tensor, TensorType,
};
use mnn_sys::HalideType;

pub type TensorCallbackT = Box<dyn Fn(&[RawTensor], OperatorInfo) -> bool>;

#[repr(transparent)]
pub struct TensorCallback {
    inner: Arc<TensorCallbackT>,
}

impl Default for TensorCallback {
    fn default() -> Self {
        Self {
            inner: Arc::new(Box::new(|_, _| true)),
        }
    }
}

impl TensorCallback {
    pub fn from_ptr(f: *mut libc::c_void) -> Self {
        debug_assert!(!f.is_null());
        unsafe {
            Self {
                inner: Arc::from_raw(f.cast()),
            }
        }
    }

    pub fn into_ptr(self) -> *mut libc::c_void {
        Arc::into_raw(self.inner) as *mut libc::c_void
    }

    pub fn identity() -> impl Fn(&[RawTensor], OperatorInfo) -> bool {
        |_, _| true
    }
}

impl<F> From<F> for TensorCallback
where
    F: Fn(&[RawTensor], OperatorInfo) -> bool + 'static,
{
    fn from(f: F) -> Self {
        Self {
            inner: Arc::new(Box::new(f)),
        }
    }
}

impl<T> From<Option<T>> for TensorCallback
where
    T: Fn(&[RawTensor], OperatorInfo) -> bool + 'static,
{
    fn from(f: Option<T>) -> Self {
        match f {
            Some(f) => Self {
                inner: Arc::new(Box::new(f)),
            },
            None => Self::default(),
        }
    }
}

impl core::ops::Deref for TensorCallback {
    type Target = TensorCallbackT;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Debug, Copy, Clone)]
#[cfg_attr(windows, repr(i32))]
#[cfg_attr(unix, repr(u32))]
pub enum SessionMode {
    #[doc = "About CallBack, Default Session_Debug*/\n/** runSessionWithCallBack is allowed and can get internal op info"]
    Debug = mnn_sys::SessionMode::Session_Debug,
    #[doc = "runSessionWithCallBack is not valid and can't get any info of op in\nsession"]
    Release = mnn_sys::SessionMode::Session_Release,
    #[doc = "About input tensor, Default Session_Input_Inside*/\n/** The input tensor is alloced by session, input data after session resized"]
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

#[cfg(windows)]
type SessionModeType = i32;
#[cfg(unix)]
type SessionModeType = u32;

impl SessionMode {
    fn to_mnn_sys(self) -> SessionModeType {
        self as SessionModeType
    }
}

/// net data holder. multiple sessions could share same net.
#[repr(transparent)]
pub struct Interpreter {
    pub(crate) inner: *mut mnn_sys::Interpreter,
    pub(crate) __marker: PhantomData<()>,
}

unsafe impl Send for Interpreter {}

impl Drop for Interpreter {
    fn drop(&mut self) {
        unsafe { mnn_sys::Interpreter_destroy(self.inner) }
    }
}

impl Interpreter {
    /// Create an net/interpreter from a file.
    ///
    /// `path`: the file path of the model
    ///
    /// return: the created net/interpreter
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        ensure!(path.exists(), ErrorKind::IOError; path.to_string_lossy().to_string(), "File not found");
        let path = path.to_str().ok_or_else(|| error!(ErrorKind::AsciiError))?;
        let c_path = std::ffi::CString::new(path).change_context(ErrorKind::AsciiError)?;
        let interpreter = unsafe { mnn_sys::Interpreter_createFromFile(c_path.as_ptr()) };
        ensure!(!interpreter.is_null(), ErrorKind::InterpreterError; "Failed to create interpreter", "Interpreter_createFromFile returned null");
        Ok(Self {
            inner: interpreter,
            __marker: PhantomData,
        })
    }

    /// Create an net/interpreter from a buffer.
    ///
    /// `bytes`: the buffer of the model
    ///
    /// return: the created net/interpreter
    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Result<Self> {
        let bytes = bytes.as_ref();
        let size = bytes.len();
        let interpreter =
            unsafe { mnn_sys::Interpreter_createFromBuffer(bytes.as_ptr().cast(), size) };
        ensure!(!interpreter.is_null(), ErrorKind::InterpreterError; "Failed to create interpreter", "Interpreter_createFromBuffer returned null");
        Ok(Self {
            inner: interpreter,
            __marker: PhantomData,
        })
    }

    /// Set session mode
    ///
    /// `mode`: the session mode
    ///
    /// **Warning:**
    /// It should be called before create session!
    pub fn set_session_mode(&mut self, mode: SessionMode) {
        unsafe { mnn_sys::Interpreter_setSessionMode(self.inner, mode.to_mnn_sys()) }
    }

    ///call this function to get tensors ready.
    ///
    ///output tensor buffer (host or deviceId) should be retrieved after resize of any input tensor.
    ///
    ///`session`: the session to be prepared
    pub fn resize_session(&self, session: &mut crate::Session) {
        unsafe { mnn_sys::Interpreter_resizeSession(self.inner, session.inner) }
    }

    /// Resize session and reallocate the buffer.
    ///
    /// `session`: the session to be prepared.
    ///
    /// # Note
    /// NeedRelloc is default to 1, 1 means need realloc!
    pub fn resize_session_reallocate(&self, session: &mut crate::Session) {
        unsafe { mnn_sys::Interpreter_resizeSessionWithFlag(self.inner, session.inner, 1i32) }
    }

    pub fn resize_tensor<T: TensorType>(&self, tensor: &mut Tensor<T>, dims: impl AsTensorShape) {
        let dims = dims.as_tensor_shape();
        let dims_len = dims.size;
        unsafe {
            mnn_sys::Interpreter_resizeTensor(
                self.inner,
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
                self.inner,
                tensor.tensor,
                batch,
                channel,
                height,
                width,
            )
        }
    }

    /// Create a session with session config. Session will be managed in net/interpreter.
    ///
    /// `schedule` : the config of the session
    ///
    /// return: the created session
    pub fn create_session(
        &mut self,
        schedule: crate::ScheduleConfig,
    ) -> Result<crate::session::Session> {
        profile!("Creating session"; {
            let session = unsafe { mnn_sys::Interpreter_createSession(self.inner, schedule.inner) };
            assert!(!session.is_null());
            Ok(crate::session::Session {
                inner: session,
                __session_internals: crate::SessionInternals::Single(schedule),
                __marker: PhantomData,
            })
        })
    }

    /// Create multi-path session with schedule configs and user-specified runtime. created session will be managed in net/interpreter.
    ///
    /// `schedule` : the config of the session
    ///
    /// return: the created session
    pub fn create_multipath_session(
        &mut self,
        schedule: impl IntoIterator<Item = ScheduleConfig>,
    ) -> Result<crate::session::Session> {
        profile!("Creating multipath session"; {
            let schedules: crate::ScheduleConfigs = schedule.into_iter().collect();
            let sc: &[_] = schedules.inner.as_ref();
            let session = unsafe { mnn_sys::Interpreter_createMultiPathSession(self.inner, sc.as_ptr(), sc.len()) };
            assert!(!session.is_null());
            Ok(crate::session::Session {
                inner: session,
                __session_internals: crate::SessionInternals::MultiSession(schedules),
                __marker: PhantomData,
            })
        })
    }

    /// Print all input and output tensors info.
    pub fn model_print_io(path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        crate::ensure!(path.exists(), ErrorKind::IOError);
        let path = path.to_str().ok_or_else(|| error!(ErrorKind::AsciiError))?;
        let c_path = std::ffi::CString::new(path).unwrap();
        unsafe { mnn_sys::modelPrintIO(c_path.as_ptr()) }
        Ok(())
    }

    /// Get the input tensor of the session.
    ///
    /// `session`: the session to get input tensor
    ///
    /// return: List of input tensors
    pub fn inputs(&self, session: &crate::Session) -> TensorList {
        let inputs = unsafe { mnn_sys::Interpreter_getSessionInputAll(self.inner, session.inner) };
        TensorList::from_ptr(inputs)
    }

    /// Get the input tensor of the session by name.
    ///
    /// `session`: the session to get input tensor from
    ///
    /// `name`: the name of the input tensor
    ///
    /// return: the input tensor
    pub fn input<'s, H: HalideType>(
        &self,
        session: &'s crate::Session,
        name: impl AsRef<str>,
    ) -> Result<Tensor<RefMut<'s, Device<H>>>> {
        let name = name.as_ref();
        let c_name = std::ffi::CString::new(name).change_context(ErrorKind::AsciiError)?;
        let input = unsafe {
            mnn_sys::Interpreter_getSessionInput(self.inner, session.inner, c_name.as_ptr())
        };
        ensure!(!input.is_null(), ErrorKind::TensorError; format!("Input tensor \"{name}\" not found"));
        let tensor = unsafe { Tensor::from_ptr(input) };
        let shape = tensor.shape();
        ensure!(!shape.as_ref().contains(&-1), ErrorKind::DynamicTensorError);
        ensure!(
            tensor.is_type_of::<H>(),
            ErrorKind::HalideTypeMismatch {
                got: std::any::type_name::<H>(),
            };
            format!("Input tensor \"{name}\" is not of type {}", std::any::type_name::<H>())
        );
        Ok(tensor)
    }

    pub fn raw_input<'s>(
        &self,
        session: &'s crate::Session,
        name: impl AsRef<str>,
    ) -> Result<RawTensor<'s>> {
        let name = name.as_ref();
        let c_name = std::ffi::CString::new(name).change_context(ErrorKind::AsciiError)?;
        let input = unsafe {
            mnn_sys::Interpreter_getSessionInput(self.inner, session.inner, c_name.as_ptr())
        };
        ensure!(!input.is_null(), ErrorKind::TensorError; format!("Input tensor \"{name}\" not found"));
        Ok(RawTensor::from_ptr(input))
    }

    /// # Safety
    /// **Warning**  We Still don't know the safety guarantees of this function so it's marked unsafe
    pub unsafe fn input_unresized<'s, H: HalideType>(
        &self,
        session: &'s crate::Session,
        name: impl AsRef<str>,
    ) -> Result<Tensor<RefMut<'s, Device<H>>>> {
        let name = name.as_ref();
        let c_name = std::ffi::CString::new(name).change_context(ErrorKind::AsciiError)?;
        let input = unsafe {
            mnn_sys::Interpreter_getSessionInput(self.inner, session.inner, c_name.as_ptr())
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

    /// # Safety
    /// Very **unsafe** since it doesn't check the type of the tensor
    /// as well as the shape of the tensor
    pub unsafe fn input_unchecked<'s, H: HalideType>(
        &self,
        session: &'s crate::Session,
        name: impl AsRef<str>,
    ) -> Tensor<RefMut<'s, Device<H>>> {
        let name = name.as_ref();
        let c_name = std::ffi::CString::new(name).unwrap();
        let input =
            mnn_sys::Interpreter_getSessionInput(self.inner, session.inner, c_name.as_ptr());
        Tensor::from_ptr(input)
    }

    /// Get the output tensor of a session by name
    ///
    /// `session` : the session to get output tensor from
    ///
    /// `name` : the name of the output tensor
    pub fn output<'s, H: HalideType>(
        &self,
        session: &'s crate::Session,
        name: impl AsRef<str>,
    ) -> Result<Tensor<Ref<'s, Device<H>>>> {
        let name = name.as_ref();
        let c_name = std::ffi::CString::new(name).change_context(ErrorKind::AsciiError)?;
        let output = unsafe {
            mnn_sys::Interpreter_getSessionOutput(self.inner, session.inner, c_name.as_ptr())
        };
        ensure!(!output.is_null(), ErrorKind::IOError;format!("Output tensor \"{name}\" not found"));
        let tensor = unsafe { Tensor::from_ptr(output) };
        let shape = tensor.shape();
        ensure!(!shape.as_ref().contains(&-1), ErrorKind::DynamicTensorError);
        ensure!(
            tensor.is_type_of::<H>(),
            ErrorKind::HalideTypeMismatch {
                got: std::any::type_name::<H>(),
            }
        );
        Ok(tensor)
    }

    pub fn raw_output<'s>(
        &self,
        session: &'s crate::Session,
        name: impl AsRef<str>,
    ) -> Result<RawTensor<'s>> {
        let name = name.as_ref();
        let c_name = std::ffi::CString::new(name).change_context(ErrorKind::AsciiError)?;
        let output = unsafe {
            mnn_sys::Interpreter_getSessionOutput(self.inner, session.inner, c_name.as_ptr())
        };
        ensure!(!output.is_null(), ErrorKind::IOError;format!("Output tensor \"{name}\" not found"));
        Ok(RawTensor::from_ptr(output))
    }

    /// Run a session
    pub fn run_session(&mut self, session: &crate::session::Session) -> Result<()> {
        profile!("Running session"; {
            let ret = unsafe { mnn_sys::Interpreter_runSession(self.inner, session.inner) };
            ensure!(
                ret == mnn_sys::ErrorCode::ERROR_CODE_NO_ERROR,
                ErrorKind::InternalError(ret)
            );
            Ok(())
        })
    }

    /// Run a session with a callback
    ///
    /// `session` : the session to run
    ///
    /// `before` : a callback before each op. return true to run the op; return false to skip the op.
    ///
    /// `after` : a callback after each op. return true to continue running; return false to interrupt the session.
    ///
    /// `sync` : synchronously wait for finish of execution or not.
    pub fn run_session_with_callback(
        &mut self,
        session: &crate::session::Session,
        before: impl Fn(&[RawTensor], OperatorInfo) -> bool + 'static,
        end: impl Fn(&[RawTensor], OperatorInfo) -> bool + 'static,
        sync: bool,
    ) -> Result<()> {
        let sync = sync as libc::c_int;
        let before = TensorCallback::from(before).into_ptr();
        let end = TensorCallback::from(end).into_ptr();
        let ret = unsafe {
            mnn_sys::Interpreter_runSessionWithCallBackInfo(
                self.inner,
                session.inner,
                before,
                end,
                sync,
            )
        };
        ensure!(
            ret == mnn_sys::ErrorCode::ERROR_CODE_NO_ERROR,
            ErrorKind::InternalError(ret)
        );
        Ok(())
    }

    /// Get all output tensors of a session
    pub fn outputs(&self, session: &crate::session::Session) -> TensorList {
        let outputs =
            unsafe { mnn_sys::Interpreter_getSessionOutputAll(self.inner, session.inner) };
        TensorList::from_ptr(outputs)
    }

    /// If the cache exist, try to load cache from file.
    /// After createSession, try to save cache to file.
    ///
    /// `cache_file` : the file path to save or load cache.
    ///
    /// `key_size` : the size of key
    ///
    /// # Note
    /// The API should be called before create session.
    ///
    /// Key Depercerate, keeping for future use!
    pub fn set_cache_file(&mut self, path: impl AsRef<Path>, key_size: usize) -> Result<()> {
        let path = path.as_ref();
        let path = dunce::simplified(path);
        let path = path.to_str().ok_or_else(|| error!(ErrorKind::AsciiError))?;
        let c_path = std::ffi::CString::new(path).change_context(ErrorKind::AsciiError)?;
        unsafe { mnn_sys::Interpreter_setCacheFile(self.inner, c_path.as_ptr(), key_size) }
        Ok(())
    }

    /// Update cache file
    pub fn update_cache_file(&mut self, session: &mut crate::session::Session) -> Result<()> {
        MNNError::from_error_code(unsafe {
            mnn_sys::Interpreter_updateCacheFile(self.inner, session.inner)
        });
        Ok(())
    }

    /// Wait for all output tensors to be ready after computation
    pub fn wait(&self, session: &crate::session::Session) {
        self.outputs(session).iter().for_each(|tinfo| {
            tinfo
                .raw_tensor()
                .wait(mnn_sys::MapType::MAP_TENSOR_READ, true);
        });
    }

    /// Get memory usage of a session in MB
    pub fn memory(&self, session: &crate::session::Session) -> Result<f32> {
        let mut memory = 0f32;
        let memory_ptr = &mut memory as *mut f32;
        let ret = unsafe {
            mnn_sys::Interpreter_getSessionInfo(self.inner, session.inner, 0, memory_ptr.cast())
        };
        ensure!(
            ret == 1,
            ErrorKind::InterpreterError;
            "Failed to get memory usage"
        );
        Ok(memory)
    }

    /// Get float operation needed in session in M
    pub fn flops(&self, session: &crate::Session) -> Result<f32> {
        let mut flop = 0.0f32;
        let flop_ptr = &mut flop as *mut f32;
        let ret = unsafe {
            mnn_sys::Interpreter_getSessionInfo(
                self.inner,
                session.inner,
                1,
                flop_ptr.cast::<libc::c_void>(),
            )
        };
        ensure!(
            ret == 1,
            ErrorKind::InterpreterError;
            "Failed to get flops"
        );
        Ok(flop)
    }

    pub fn resize_status(&self, session: &crate::Session) -> Result<ResizeStatus> {
        let mut resize_status = 0i32;
        let ptr = &mut resize_status as *mut i32;
        let ret = unsafe {
            mnn_sys::Interpreter_getSessionInfo(self.inner, session.inner, 3, ptr.cast())
        };
        ensure!(
        ret == 1,
            ErrorKind::InterpreterError;
            "Failed to get resize status"
        );
        match resize_status {
            0 => Ok(ResizeStatus::None),
            1 => Ok(ResizeStatus::NeedMalloc),
            2 => Ok(ResizeStatus::NeedResize),
            _ => Err(error!(ErrorKind::InterpreterError)),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(C)]
pub enum ResizeStatus {
    None = 0,
    NeedMalloc = 1,
    NeedResize = 2,
}

#[repr(transparent)]
pub struct TensorInfo<'t, 'tl> {
    pub(crate) tensor_info: *mut mnn_sys::TensorInfo,
    pub(crate) __marker: PhantomData<&'tl TensorList<'t>>,
}

impl core::fmt::Debug for TensorInfo<'_, '_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let tensor = self.raw_tensor();
        let shape = tensor.shape().clone();
        f.debug_struct("TensorInfo")
            .field("name", &self.name())
            .field("tensor", &shape)
            .finish()
    }
}

impl<'t, 'tl> TensorInfo<'t, 'tl> {
    pub fn name(&self) -> &'tl str {
        debug_assert!(!self.tensor_info.is_null());
        unsafe { (*self.tensor_info).name.to_cstr() }
            .to_str()
            .expect("Tensor name is not utf-8")
    }

    pub fn tensor<H: HalideType>(&self) -> Result<Tensor<RefMut<'t, Device<H>>>> {
        debug_assert!(!self.tensor_info.is_null());
        unsafe { debug_assert!(!(*self.tensor_info).tensor.is_null()) };
        let tensor = unsafe { Tensor::from_ptr((*self.tensor_info).tensor.cast()) };
        let shape = tensor.shape();
        ensure!(!shape.as_ref().contains(&-1), ErrorKind::DynamicTensorError);
        ensure!(
            tensor.is_type_of::<H>(),
            ErrorKind::HalideTypeMismatch {
                got: std::any::type_name::<H>(),
            }
        );
        Ok(tensor)
    }

    /// # Safety
    /// The shape is not checked so it's marked unsafe since futher calls to interpreter might be **unsafe** with this
    pub unsafe fn tensor_unresized<H: HalideType>(&self) -> Result<Tensor<RefMut<'t, Device<H>>>> {
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

    pub fn raw_tensor(&self) -> RawTensor<'t> {
        debug_assert!(!self.tensor_info.is_null());
        unsafe { debug_assert!(!(*self.tensor_info).tensor.is_null()) };
        RawTensor::from_ptr(unsafe { (*self.tensor_info).tensor.cast() })
    }
}

#[repr(transparent)]
pub struct TensorList<'t> {
    pub(crate) inner: *const mnn_sys::TensorInfoArray,
    pub(crate) __marker: PhantomData<&'t Interpreter>,
}

impl core::fmt::Debug for TensorList<'_> {
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

    pub fn size(&self) -> usize {
        unsafe { (*self.inner).size }
    }

    pub fn get(&self, index: usize) -> Option<TensorInfo<'t, '_>> {
        if index >= self.size() {
            None
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

impl<'t, 'tl: 't> IntoIterator for &'tl TensorList<'t> {
    type Item = TensorInfo<'t, 'tl>;
    type IntoIter = TensorListIter<'t, 'tl>;

    fn into_iter(self) -> Self::IntoIter {
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

// #[no_mangle]
// extern "C" fn rust_closure_callback_runner(
//     f: *mut libc::c_void,
//     tensors: *const *mut mnn_sys::Tensor,
//     tensor_count: usize,
//     name: *const libc::c_char,
// ) -> libc::c_int {
//     let tensors = unsafe { std::slice::from_raw_parts(tensors.cast(), tensor_count) };
//     let name = unsafe { std::ffi::CStr::from_ptr(name) };
//     let f: TensorCallback = unsafe { Box::from_raw(f.cast::<TensorCallback>()) };
//     let ret = f(tensors, name) as libc::c_int;
//     core::mem::forget(f);
//     ret
// }

// #[test]
// fn test_extern_c_rust_closure_callback_runner() {
//     let f = |_tensors: &[RawTensor], name: &CStr| -> bool {
//         println!("Callback: {:?}", name);
//         true
//     };
//     let f: Box<TensorCallback> = Box::new(Box::new(f));
//     let f = Box::into_raw(f).cast();
//     let tensors = [std::ptr::null_mut()];
//     let name = std::ffi::CString::new("Test").unwrap();
//     let ret = rust_closure_callback_runner(f, tensors.as_ptr(), tensors.len(), name.as_ptr())
//         as libc::c_int;
//     assert_eq!(ret, 0);
// }

#[no_mangle]
extern "C" fn rust_closure_callback_runner_op(
    f: *mut libc::c_void,
    tensors: *const *mut mnn_sys::Tensor,
    tensor_count: usize,
    op: *mut libc::c_void,
) -> libc::c_int {
    let tensors = unsafe { std::slice::from_raw_parts(tensors.cast(), tensor_count) };
    let f: TensorCallback = TensorCallback::from_ptr(f);
    let op = OperatorInfo {
        inner: op.cast(),
        __marker: PhantomData,
    };
    let ret = f(tensors, op) as libc::c_int;

    core::mem::forget(f);
    ret
}

#[repr(transparent)]
pub struct OperatorInfo<'op> {
    pub(crate) inner: *mut libc::c_void,
    pub(crate) __marker: PhantomData<&'op ()>,
}

impl core::fmt::Debug for OperatorInfo<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OperatorInfo")
            .field("name", &self.name())
            .field("type", &self.type_name())
            .field("flops", &self.flops())
            .finish()
    }
}

impl OperatorInfo<'_> {
    pub fn name(&self) -> &CStr {
        unsafe { CStr::from_ptr(mnn_sys::OperatorInfo_name(self.inner)) }
    }

    pub fn type_name(&self) -> &CStr {
        unsafe { CStr::from_ptr(mnn_sys::OperatorInfo_type(self.inner)) }
    }

    pub fn flops(&self) -> f32 {
        unsafe { mnn_sys::OperatorInfo_flops(self.inner) }
    }
}

#[test]
#[ignore = "This test doesn't work in CI"]
fn test_run_session_with_callback_info_api() {
    let file = Path::new("tests/assets/realesr.mnn")
        .canonicalize()
        .unwrap();
    let mut interpreter = Interpreter::from_file(&file).unwrap();
    let session = interpreter.create_session(ScheduleConfig::new()).unwrap();
    interpreter
        .run_session_with_callback(
            &session,
            TensorCallback::identity(),
            TensorCallback::identity(),
            true,
        )
        .unwrap();
}

#[test]
#[ignore = "This test doesn't work in CI"]
fn check_whether_sync_actually_works() {
    let file = Path::new("tests/assets/realesr.mnn")
        .canonicalize()
        .unwrap();
    let mut interpreter = Interpreter::from_file(&file).unwrap();
    let session = interpreter.create_session(ScheduleConfig::new()).unwrap();
    let time = std::time::Instant::now();
    interpreter
        .run_session_with_callback(
            &session,
            TensorCallback::identity(),
            TensorCallback::identity(),
            false,
        )
        .unwrap();
    let time = time.elapsed();
    let time2 = std::time::Instant::now();
    interpreter
        .run_session_with_callback(
            &session,
            TensorCallback::identity(),
            TensorCallback::identity(),
            true,
        )
        .unwrap();
    let time2 = time2.elapsed();
    assert!((time - time2) > std::time::Duration::from_millis(50));
}

#[test]
#[ignore = "This test doesn't work in CI"]
fn try_to_drop_interpreter_before_session() {
    let file = Path::new("tests/assets/realesr.mnn")
        .canonicalize()
        .unwrap();
    let mut interpreter = Interpreter::from_file(&file).unwrap();
    let session = interpreter.create_session(ScheduleConfig::new()).unwrap();
    drop(interpreter);
    drop(session);
}
