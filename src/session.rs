use crate::prelude::*;

/// A session is a context in which a computation graph is executed.
///
/// Inference unit. multiple sessions could share one net/interpreter.
#[derive(Debug)]
pub struct Session<'i> {
    /// Pointer to the underlying MNN session.
    pub(crate) inner: *mut mnn_sys::Session,
    /// Pointer to the underlying MNN interpreter
    /// # Safety Note
    /// Since the interpreter is actually not owned by session but it is a shared resource we can
    /// reasonably assume that the interpreter will outlive the session. (This is not a compile
    /// time gurantee yet)
    /// TODO: Add a proper lifetime bound to ensure the interpreter outlives the session.
    pub(crate) net: *mut mnn_sys::Interpreter,
    /// Internal session configurations.
    pub(crate) __session_internals: crate::SessionInternals,
    /// Marker to ensure the struct is not Send or Sync.
    pub(crate) __marker: PhantomData<&'i ()>,
}

/// Enum representing the internal configurations of a session.
#[derive(Debug)]
pub enum SessionInternals {
    /// Single session configuration.
    Single(crate::ScheduleConfig),
    /// Multiple session configurations.
    MultiSession(crate::ScheduleConfigs),
}

impl Session<'_> {
    /// Calls the destructor for the underlying MNN session.
    pub fn destroy(&mut self) {
        unsafe {
            mnn_sys::Interpreter_releaseSession(self.net, self.inner);
        }
        // unsafe { mnn_sys::Session_destroy(self.inner) }
    }
}

impl Drop for Session<'_> {
    /// Custom drop implementation to ensure the underlying MNN session is properly destroyed.
    fn drop(&mut self) {
        self.destroy();
    }
}
