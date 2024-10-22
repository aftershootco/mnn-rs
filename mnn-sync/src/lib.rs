//! Synchronous API for MNN
//! This api allows use of mnn in a thread-safe manner  
//! # Example
//! ```rust,no_run
//! use mnn_sync::*;
//! use mnn::*;
//! let interpreter = Interpreter::from_bytes([0; 100]).expect("Failed to create interpreter");
//! let config = ScheduleConfig::new();
//! let session_handle = SessionHandle::new(interpreter, config).expect("Failed to create session handle");
//! std::thread::spawn(move || {
//!     session_handle.run(|sr| {
//!         let session = sr.session();
//!         let interpreter = sr.interpreter();
//!         let mut input = interpreter.input::<f32>(session, "input")?;
//!         input.fill(1.0f32);
//!         Ok(())
//!     }).expect("Failed to run");
//!     session_handle.run(|sr| {
//!         sr.run_session()?;
//!         Ok(())
//!     }).expect("Failed to run");
//! });
//! ```
//! ## Architecture
//! This is achieved by creating a [std::thread::Thread] that creates a [Session] and takes [FnOnce] through a  
//!
//! [std::sync::mpsc] channel and runs them in the [Session].
//!
//! The [Session] is closed when the [SessionHandle] is dropped.  
//!
//! The following is a diagram of the architecture of the sync api  
#![doc = "<div align=''>\n"]
#![doc = include_str!("../../docs/assets/mnn-architecture.svg")]
#![doc = "</div>\n"]
//! When you run a closure it is sent to the thread and executed in that session and the result is  
//! sent back to the main thread via a [oneshot::Sender]

use flume::{Receiver, Sender};

use error_stack::{Report, ResultExt};
use mnn::*;

type Callback = Box<dyn FnOnce(&mut SessionRunner) -> Result<()> + Send + 'static>;
pub enum CallbackEnum {
    Callback(Callback),
    Close,
}
// type CallbackSender = (CallbackEnum, oneshot::Sender<Result<()>>);
type CallbackSender = CallbackEnum;

#[derive(Debug)]
pub struct SessionHandle {
    #[allow(dead_code)]
    pub(crate) handle: std::thread::JoinHandle<Result<()>>,
    pub(crate) sender: Sender<CallbackSender>,
    pub(crate) loop_handle: Receiver<Result<()>>,
}

impl Drop for SessionHandle {
    fn drop(&mut self) {
        self.sender
            .send(CallbackEnum::Close)
            .expect("Failed to close SessionHandle");
        // rx.recv().expect("Failed to close SessionHandle");
    }
}

#[non_exhaustive]
#[derive(Debug)]
pub struct SessionRunner {
    pub interpreter: Interpreter,
    pub session: Session,
}

impl SessionHandle {
    pub fn new(mut interpreter: Interpreter, config: ScheduleConfig) -> Result<Self> {
        let (sender, receiver) = flume::unbounded::<CallbackSender>();

        let builder = std::thread::Builder::new().name("mnn-session-thread".to_string());
        let (tx, rx) = flume::unbounded();
        let handle = builder
            .spawn(move || -> Result<()> {
                #[cfg(feature = "tracing")]
                tracing::trace!("Initializing mnn session thread");
                let mut session = interpreter.create_session(config)?;
                #[cfg(feature = "tracing")]
                tracing::trace!("Updating mnn cache file");
                interpreter.update_cache_file(&mut session)?;
                let mut session_runner = SessionRunner {
                    interpreter,
                    session,
                };
                #[cfg(feature = "tracing")]
                tracing::trace!("Initializing mnn session loop");
                loop {
                    let f = receiver
                        .recv()
                        .change_context(ErrorKind::SyncError)
                        .attach_printable("Internal Error: Unable to recv (Sender Dropped)")?;
                    let f = match f {
                        CallbackEnum::Callback(f) => f,
                        CallbackEnum::Close => break,
                    };
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        f(&mut session_runner)
                    }))
                    .unwrap_or_else(|e| {
                        let mut err =
                            Report::new(ErrorKind::SyncError).attach_printable(format!("{:?}", e));
                        if let Some(location) = e.downcast_ref::<core::panic::Location>() {
                            err = err.attach_printable(format!("{:?}", location));
                        };
                        if let Some(backtrace) = e.downcast_ref::<std::backtrace::Backtrace>() {
                            err = err.attach_printable(format!("{:?}", backtrace));
                        };
                        let ret = Err(MNNError::from(err));
                        #[cfg(feature = "tracing")]
                        tracing::error!("Panic in session thread: {:?}", ret);
                        ret
                    });
                    tx.send(result)
                        .change_context(ErrorKind::SyncError)
                        .attach_printable(
                            "Internal Error: Failed to send result via oneshot channel",
                        )?;
                }
                Ok(())
            })
            .change_context(ErrorKind::SyncError)
            .attach_printable("Internal Error: Failed to create session thread")?;
        // rx.recv()
        //     .change_context(ErrorKind::SyncError)
        //     .attach_printable("Internal Error: Unable to recv message")??;

        Ok(Self {
            handle,
            sender,
            loop_handle: rx,
        })
    }

    pub fn run<R: Send + Sync + 'static>(
        &self,
        f: impl FnOnce(&mut SessionRunner) -> Result<R> + Send + Sync + 'static,
    ) -> Result<R> {
        let f = f;
        let (tx, rx) = oneshot::channel();
        let wrapped_f = move |sr: &mut SessionRunner| -> Result<()> {
            let result = f(sr);
            tx.send(result)
                .change_context(ErrorKind::SyncError)
                .attach_printable("Internal Error: Failed to send result via oneshot channel")?;
            Ok(())
        };
        self.sender
            .send(CallbackEnum::Callback(Box::new(wrapped_f)))
            .map_err(|e| Report::new(ErrorKind::SyncError).attach_printable(e.to_string()))?;
        Ok(rx
            .recv()
            .change_context(ErrorKind::SyncError)
            .attach_printable("Internal Error: Unable to recv message")??)
    }

    pub async fn run_async<R: Send + Sync + 'static>(
        &self,
        f: impl FnOnce(&mut SessionRunner) -> Result<R> + Send + Sync + 'static,
    ) -> Result<R> {
        let f = f;
        let (tx, rx) = oneshot::channel();
        let wrapped_f = move |sr: &mut SessionRunner| -> Result<()> {
            let result = f(sr)?;
            tx.send(result)
                .change_context(ErrorKind::SyncError)
                .attach_printable("Internal Error: Failed to send result via oneshot channel")?;
            Ok(())
        };
        self.sender
            .send(CallbackEnum::Callback(Box::new(wrapped_f)))
            .map_err(|e| Report::new(ErrorKind::SyncError).attach_printable(e.to_string()))?;
        Ok(rx
            .await
            .change_context(ErrorKind::SyncError)
            .attach_printable("Internal Error: Unable to recv message")?)
    }

    pub fn panicked(&self) -> bool {
        self.loop_handle
            .try_recv()
            .map(|p| p.is_err())
            .unwrap_or(false)
    }
}

impl SessionRunner {
    pub fn run_session(&mut self) -> Result<()> {
        self.interpreter.run_session(&self.session)
    }

    pub fn resize_session(&mut self) -> Result<()> {
        self.interpreter.resize_session(&mut self.session);
        Ok(())
    }

    pub fn interpreter(&self) -> &Interpreter {
        &self.interpreter
    }

    pub fn interpreter_mut(&mut self) -> &mut Interpreter {
        &mut self.interpreter
    }

    pub fn session(&self) -> &Session {
        &self.session
    }
    pub fn session_mut(&mut self) -> &mut Session {
        &mut self.session
    }
}

#[test]
#[should_panic]
pub fn test_sync_api() {
    let interpreter = Interpreter::from_bytes([0; 100]).expect("Failed to create interpreter");
    let config = ScheduleConfig::new();
    let session_handle =
        SessionHandle::new(interpreter, config).expect("Failed to create session handle");
    let my_arr = [1f32; 100];
    session_handle
        .run(move |sr| {
            let session = sr.session();
            let interpreter = sr.interpreter();
            let mut input = interpreter.input::<f32>(session, "input")?;
            let mut cpu_input = input.create_host_tensor_from_device(false);
            cpu_input.host_mut().copy_from_slice(&my_arr);
            input.copy_from_host_tensor(&cpu_input)?;
            Ok(())
        })
        .expect("Failed to run");

    session_handle
        .run(|sr| {
            sr.run_session()?;
            Ok(())
        })
        .expect("Failed to run");

    session_handle
        .run(|sr| {
            let output = sr.interpreter().output::<f32>(sr.session(), "output")?;
            let cpu_output = output.create_host_tensor_from_device(true);
            cpu_output.host().to_vec();
            Ok(())
        })
        .expect("Sed");
}

#[test]
#[ignore = "This test is not reliable on CI"]
pub fn test_sync_api_race() {
    let interpreter =
        Interpreter::from_file("tests/assets/realesr.mnn").expect("Failed to create interpreter");
    let session_handle = SessionHandle::new(interpreter, ScheduleConfig::new())
        .expect("Failed to create session handle");
    session_handle
        .run(move |sr| {
            let session = sr.session();
            let interpreter = sr.interpreter();
            let inputs = interpreter.inputs(session);
            inputs.iter().for_each(|x| {
                let mut tensor = x.tensor::<f32>().expect("No tensor");
                println!("{}: {:?}", x.name(), tensor.shape());
                let mut cpu_tensor = tensor.create_host_tensor_from_device(false);
                cpu_tensor.host_mut().fill(1.0f32);
                tensor
                    .copy_from_host_tensor(&cpu_tensor)
                    .expect("Could not copy tensor");
            });
            Ok(())
        })
        .expect("Failed to run");

    session_handle
        .run(|sr| {
            sr.run_session()?;
            Ok(())
        })
        .expect("Failed to run");
    session_handle
        .run(move |sr| {
            let session = sr.session();
            let interpreter = sr.interpreter();
            let inputs = interpreter.inputs(session);
            inputs.iter().for_each(|x| {
                let mut tensor = x.tensor::<f32>().expect("No tensor");
                println!("{}: {:?}", x.name(), tensor.shape());
                let mut cpu_tensor = tensor.create_host_tensor_from_device(false);
                cpu_tensor.host_mut().fill(1.0f32);
                tensor
                    .copy_from_host_tensor(&cpu_tensor)
                    .expect("Could not copy tensor");
            });
            Ok(())
        })
        .expect("Failed to run");
    session_handle
        .run(move |sr| {
            let session = sr.session();
            let interpreter = sr.interpreter();
            let inputs = interpreter.inputs(session);
            inputs.iter().for_each(|x| {
                let mut tensor = x.tensor::<f32>().expect("No tensor");
                println!("{}: {:?}", x.name(), tensor.shape());
                let mut cpu_tensor = tensor.create_host_tensor_from_device(false);
                cpu_tensor.host_mut().fill(1.0f32);
                tensor
                    .copy_from_host_tensor(&cpu_tensor)
                    .expect("Could not copy tensor");
            });
            Ok(())
        })
        .expect("Failed to run");
    session_handle
        .run(|sr| {
            sr.run_session()?;
            Ok(())
        })
        .expect("Failed to run");

    let _vec: Vec<f32> = session_handle
        .run(|sr| {
            let output = sr.interpreter().output::<f32>(sr.session(), "output")?;
            let cpu_output = output.create_host_tensor_from_device(true);
            Ok(cpu_output.host().to_vec())
        })
        .expect("Sed");
}

#[test]
pub fn test_sync_api_is_send_sync() {
    fn is_send_sync<T: Send + Sync>() {}
    is_send_sync::<SessionHandle>();
}
