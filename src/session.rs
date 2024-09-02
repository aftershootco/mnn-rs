use crate::prelude::*;

pub struct Session {
    pub(crate) inner: *mut mnn_sys::Session,
    // pub(crate) backend_config: BackendConfig,
    pub(crate) schedule_config: crate::ScheduleConfig,
    pub(crate) __marker: PhantomData<()>,
}

impl Session {
    // pub unsafe fn from_ptr(session: *mut mnn_sys::Session) -> Self {
    //     Self {
    //         session,
    //         __marker: PhantomData,
    //     }
    // }

    // pub fn as_ptr_mut(&self) -> *mut mnn_sys::Session {
    //     self.session
    // }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { mnn_sys::Session_destroy(self.inner) }
    }
}

// pub struct SessionInterpreter {
//     session: Session<'static>,
//     interpreter: Interpreter,
// }
