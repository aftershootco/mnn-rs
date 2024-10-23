use crate::prelude::*;

/// A session is a context in which a computation graph is executed.
///
/// Inference unit. multiple sessions could share one net/interpreter.
pub struct Session {
    /// Pointer to the underlying MNN session.
    pub(crate) inner: *mut mnn_sys::Session,
    /// Internal session configurations.
    pub(crate) __session_internals: crate::SessionInternals,
    /// Marker to ensure the struct is not Send or Sync.
    pub(crate) __marker: PhantomData<()>,
}

/// Enum representing the internal configurations of a session.
pub enum SessionInternals {
    /// Single session configuration.
    Single(crate::ScheduleConfig),
    /// Multiple session configurations.
    MultiSession(crate::ScheduleConfigs),
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
    /// Custom drop implementation to ensure the underlying MNN session is properly destroyed.
    fn drop(&mut self) {
        unsafe { mnn_sys::Session_destroy(self.inner) }
    }
}
