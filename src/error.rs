use mnn_sys::ErrorCode;

pub type Result<T, E = MNNError> = core::result::Result<T, E>;

pub struct MNNError {
    kind: error_stack::Report<ErrorKind>,
}

impl MNNError {
    pub fn into_report(self) -> error_stack::Report<ErrorKind> {
        self.kind
    }
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

#[derive(thiserror::Error, Debug)]
pub enum ErrorKind {
    #[error("Internal error: {0:?}")]
    InternalError(ErrorCode),
    #[error("Invalid input: expected {expected}, got {got}")]
    SizeMismatch { expected: usize, got: usize },
    #[error("Failed to copy tensor")]
    TensorCopyFailed(i32),
    #[error("IO Error")]
    IOError,
    #[error("Interpreter Error")]
    InterpreterError,
    #[error("Ascii Error")]
    AsciiError,
    #[error("HalideType mismatch: got {got}")]
    HalideTypeMismatch { got: &'static str },
    #[error("Parse Error")]
    ParseError,
    #[error("Sync Error")]
    SyncError,
    #[error("Tensor Error")]
    TensorError,
    #[error("Dynamic Tensor Error: Tensor needs to be resized before using")]
    DynamicTensorError,
}

impl MNNError {
    #[track_caller]
    pub fn new(kind: ErrorKind) -> Self {
        let kind = error_stack::Report::new(kind);
        Self { kind }
    }

    #[track_caller]
    pub fn from_error_code(code: ErrorCode) -> Self {
        Self::new(ErrorKind::InternalError(code))
    }

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
    pub fn attach_printable(
        self,
        printable: impl core::fmt::Display + core::fmt::Debug + Send + Sync + 'static,
    ) -> Self {
        let kind = self.kind.attach_printable(printable);
        Self { kind }
    }
}
