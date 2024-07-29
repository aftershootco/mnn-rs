use mnn_sys::ErrorCode;

pub type Result<T, E = MNNError> = core::result::Result<T, E>;
pub struct MNNError {
    #[cfg(feature = "error-report")]
    kind: error_stack::Report<ErrorKind>,
    #[cfg(not(feature = "error-report"))]
    kind: ErrorKind,
}

impl core::fmt::Display for MNNError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.kind)
    }
}

impl core::fmt::Debug for MNNError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self.kind)
    }
}

impl std::error::Error for MNNError {}

#[derive(thiserror::Error, Debug)]
pub enum ErrorKind {
    #[error("Internal error: {0:?}")]
    InternalError(ErrorCode),
    #[error("Invalid input: expected {expected}, got {got}")]
    SizeMismatch { expected: usize, got: usize },
    #[error("Failed to copy tonsor")]
    TensorCopyFailed,
}

impl MNNError {
    #[track_caller]
    pub fn new(kind: ErrorKind) -> Self {
        #[cfg(feature = "error-report")]
        let kind = error_stack::Report::new(kind);
        Self { kind }
    }

    #[track_caller]
    pub fn from_error_code(code: ErrorCode) -> Self {
        Self::new(ErrorKind::InternalError(code))
    }
}

impl From<ErrorKind> for MNNError {
    #[track_caller]
    fn from(kind: ErrorKind) -> Self {
        Self::new(kind)
    }
}

macro_rules! ensure {
    ($cond:expr, $kind:expr) => {
        if !$cond {
            return Err(crate::error::MNNError::new($kind));
        }
    };
}

pub(crate) use ensure;
