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
        write!(f, "{:?}", self.kind)
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
    #[error("Failed to copy tensor")]
    TensorCopyFailed,
    #[error("IO Error")]
    IOError,
    #[error("Interpreter Error")]
    InterpreterError,
    #[error("Ascii Error")]
    AsciiError,
    #[error("HalideType mismatch: got {got}")]
    HalideTypeMismatch { got: &'static str },
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
        if !($cond) {
            return Err(crate::error::MNNError::new($kind));
        }
    };
    ($cond:expr, $from:expr, $to:expr) => {
        if (!$cond) {
            #[cfg(feature = "error-report")]
            return Err(error_stack::Report::new($from).change_context($to));
            #[cfg(not(feature = "error-report"))]
            return Err(crate::error::MNNError::new($to));
        }
    };
}

macro_rules! error {
    ($kind:expr) => {
        crate::error::MNNError::new($kind)
    };
    ($kind:expr, $from:expr) => {
        crate::error::MNNError::new(error_stack::Report::new($from).change_context($kind))
    };
}

pub(crate) use ensure;
pub(crate) use error;

impl From<error_stack::Report<ErrorKind>> for MNNError {
    #[track_caller]
    fn from(report: error_stack::Report<ErrorKind>) -> Self {
        Self { kind: report }
    }
}

impl MNNError {
    pub fn attach_printable(
        self,
        printable: impl core::fmt::Display + core::fmt::Debug + Send + Sync + 'static,
    ) -> Self {
        #[cfg(feature = "error-report")]
        let kind = self.kind.attach_printable(printable);
        #[cfg(not(feature = "error-report"))]
        let kind = self.kind;
        Self { kind }
    }
}
