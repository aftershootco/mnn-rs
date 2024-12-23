use mnn_sys::ErrorCode;

#[doc(hidden)]
pub type Result<T, E = MNNError> = core::result::Result<T, E>;

/// Error type container for MNN
pub struct MNNError {
    kind: error_stack::Report<ErrorKind>,
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
// pub type MNNError = error_stack::Report<ErrorKind>;

/// Error types for MNN
#[derive(thiserror::Error, Debug)]
pub enum ErrorKind {
    /// Internal error (from MNN library)
    #[error("Internal error: {0:?}")]
    InternalError(ErrorCode),
    /// Mismatching Size for input
    #[error("Invalid input: expected {expected}, got {got}")]
    SizeMismatch {
        /// Expected size
        expected: usize,
        /// Provided size
        got: usize,
    },
    /// Failed to copy tensor
    #[error("Failed to copy tensor")]
    TensorCopyFailed(i32),
    /// I/O Error
    #[error("IO Error")]
    IOError,
    /// Interpreter Error
    #[error("Interpreter Error")]
    InterpreterError,
    /// ASCII Error (path, name, etc had invalid characters)
    #[error("Ascii Error")]
    AsciiError,
    /// HalideType mismatch (e.g. trying to convert from a float tensor to an int tensor)
    #[error("HalideType mismatch: got {got}")]
    HalideTypeMismatch {
        /// HalideType that was
        got: &'static str,
    },
    /// Failed to parse the Argument
    #[error("Parse Error")]
    ParseError,
    /// Error with mnn-sync crate
    #[error("Sync Error")]
    SyncError,
    /// Error with some tensor
    #[error("Tensor Error")]
    TensorError,
    /// Tried to run a dynamic tensor without resizing it first
    #[error("Dynamic Tensor Error: Tensor needs to be resized before using")]
    DynamicTensorError,
}

impl MNNError {
    #[track_caller]
    #[doc(hidden)]
    pub fn new(kind: ErrorKind) -> Self {
        let kind = error_stack::Report::new(kind);
        Self { kind }
    }

    #[track_caller]
    pub(crate) fn from_error_code(code: ErrorCode) -> Self {
        Self::new(ErrorKind::InternalError(code))
    }

    /// Return the inner [error_stack::Report] containing the error
    #[inline(always)]
    pub fn into_inner(self) -> error_stack::Report<ErrorKind> {
        self.kind
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

    ($cond:expr, $kind:expr; $($printable:expr),*) => {
        if !($cond) {
            return Err(crate::error::MNNError::new($kind)
                $(.attach_printable($printable))*
            )
        }
    };


    ($cond:expr, $from:expr, $to:expr) => {
        if (!$cond) {
            return Err(error_stack::Report::new($from).change_context($to));
        }
    };
    ($cond:expr, $from:expr, $to:expr; $($printable:expr),*) => {
        if (!$cond) {
            return Err(error_stack::Report::new($from)
                .change_context($to)
                $(.attach_printable($printable))*
            )
        }
    };
}

macro_rules! error {
    ($kind:expr) => {
        crate::error::MNNError::new($kind)
    };
    ($kind:expr, $from:expr) => {
        crate::error::MNNError::from(error_stack::Report::new($from).change_context($kind))
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
    pub(crate) fn attach_printable(
        self,
        printable: impl core::fmt::Display + core::fmt::Debug + Send + Sync + 'static,
    ) -> Self {
        let kind = self.kind.attach_printable(printable);
        Self { kind }
    }
}
