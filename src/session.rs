use crate::prelude::*;

pub struct Session {
    pub(crate) inner: *mut mnn_sys::Session,
    pub(crate) __session_internals: crate::SessionInternals,
    pub(crate) __marker: PhantomData<()>,
}

pub enum SessionInternals {
    Single(crate::ScheduleConfig),
    MultiSession(crate::ScheduleConfigs),
}

impl Session {
    // pub unsafe fn from_ptr(session: *mut mnn_sys::Session) -> Self {
    //     Self {
    //         session,
    //         __marker: PhantomData,
    //     }
    // }
    pub fn has_async_work(&self) -> bool {
        unsafe { mnn_sys::Session_hasAsyncWork(self.inner) != 0 }
    }
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
