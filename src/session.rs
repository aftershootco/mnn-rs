use core::marker::PhantomData;

pub struct Session {
    pub(crate) session: *mut mnn_sys::Session,
    pub(crate) __marker: PhantomData<()>,
}

impl Session {
    pub unsafe fn from_ptr(session: *mut mnn_sys::Session) -> Self {
        Self {
            session,
            __marker: PhantomData,
        }
    }
    pub fn raw_mut(&self) -> *mut mnn_sys::Session {
        self.session
    }
}
