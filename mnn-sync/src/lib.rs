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

use flume::{Receiver, Sender};

use error_stack::{Report, ResultExt};
use mnn::*;

type Callback = Box<dyn FnOnce(&mut SessionRunner) -> Result<()> + Send + 'static>;

pub enum CallbackEnum {
    Callback(Callback),
    Unload(oneshot::Sender<Result<()>>),
    Load(oneshot::Sender<Result<()>>),
    Status(oneshot::Sender<bool>),
    Close,
}
type CallbackSender = CallbackEnum;

#[derive(Debug)]
pub struct SessionHandle {
    #[allow(dead_code)]
    pub(crate) handle: Option<std::thread::JoinHandle<Result<()>>>,
    pub(crate) sender: Sender<CallbackSender>,
}

impl Drop for SessionHandle {
    fn drop(&mut self) {
        #[cfg(feature = "tracing")]
        tracing::info!("Dropping SessionHandle");
        self.close().expect("Failed to close session");
        self.handle
            .take()
            .map(|j| j.join().expect("Failed to join thread"));
    }
}

#[derive(Debug)]
pub struct SessionState {
    sr: SessionRunnerState,
    receiver: Receiver<CallbackSender>,
    config: ScheduleConfig,
}

#[derive(Debug, Default)]
pub enum SessionRunnerState {
    Loaded(SessionRunner),
    Unloaded(Interpreter),
    #[default]
    Poisoned,
}

impl SessionRunnerState {
    pub fn is_loaded(&self) -> bool {
        matches!(self, SessionRunnerState::Loaded(_))
    }

    pub fn is_unloaded(&self) -> bool {
        matches!(self, SessionRunnerState::Unloaded(_))
    }

    pub fn is_poisoned(&self) -> bool {
        matches!(self, SessionRunnerState::Poisoned)
    }

    pub fn loaded(&self) -> Option<&SessionRunner> {
        match self {
            Self::Loaded(sr) => Some(sr),
            _ => None,
        }
    }

    pub fn unloaded(&self) -> Option<&Interpreter> {
        match self {
            Self::Unloaded(net) => Some(net),
            _ => None,
        }
    }

    pub fn loaded_mut(&mut self) -> Option<&mut SessionRunner> {
        match self {
            Self::Loaded(sr) => Some(sr),
            _ => None,
        }
    }

    pub fn unloaded_mut(&mut self) -> Option<&mut Interpreter> {
        match self {
            Self::Unloaded(net) => Some(net),
            _ => None,
        }
    }

    pub fn unload(&mut self) -> Result<()> {
        #[cfg(feature = "tracing")]
        tracing::trace!("Unloading session");
        match core::mem::take(self) {
            Self::Loaded(sr) => {
                let net = sr.unload()?;
                *self = Self::Unloaded(net);
                Ok(())
            }
            Self::Unloaded(u) => {
                *self = Self::Unloaded(u);
                Ok(())
            }
            Self::Poisoned => Self::poisoned(),
        }
    }

    pub fn load(&mut self, config: &ScheduleConfig) -> Result<()> {
        #[cfg(feature = "tracing")]
        tracing::trace!("Loading session");
        match core::mem::take(self) {
            Self::Loaded(sr) => {
                *self = Self::Loaded(sr);
                Ok(())
            }
            Self::Unloaded(net) => {
                let sr = SessionRunner::create(net, config.clone())?;
                *self = Self::Loaded(sr);
                Ok(())
            }
            Self::Poisoned => Self::poisoned(),
        }
    }

    pub fn sr(&mut self, config: &ScheduleConfig) -> Result<&mut SessionRunner> {
        match self {
            Self::Loaded(sr) => Ok(sr),
            Self::Unloaded(_) => {
                self.load(config)?;
                Ok(self.loaded_mut().ok_or_else(|| {
                    Report::new(ErrorKind::SyncError).attach_printable("Failed to load session")
                })?)
            }
            Self::Poisoned => {
                Err(Report::new(ErrorKind::SyncError).attach_printable("Poisoned Session"))?
            }
        }
    }

    fn poisoned() -> Result<()> {
        Err(Report::new(ErrorKind::SyncError).attach_printable("Poisoned Session"))?;
        Ok(())
    }
}

impl SessionState {
    pub fn sr(&mut self) -> Result<&mut SessionRunner> {
        self.sr.sr(&self.config)
    }

    pub fn load(&mut self) -> Result<()> {
        self.sr.load(&self.config)
    }

    pub fn unload(&mut self) -> Result<()> {
        self.sr.unload()
    }

    pub fn is_loaded(&self) -> bool {
        self.sr.is_loaded()
    }
}

#[non_exhaustive]
#[derive(Debug)]
pub struct SessionRunner {
    pub interpreter: Interpreter,
    pub session: Session,
}

impl SessionRunner {
    pub fn new(interpreter: Interpreter, session: Session) -> Self {
        Self {
            interpreter,
            session,
        }
    }

    pub fn create(mut net: Interpreter, config: ScheduleConfig) -> Result<Self> {
        #[cfg(feature = "tracing")]
        tracing::trace!("Creating session");
        #[cfg(feature = "tracing")]
        let now = std::time::Instant::now();
        let mut session = net.create_session(config)?;
        net.update_cache_file(&mut session)?;
        #[cfg(feature = "tracing")]
        tracing::trace!("Session created in {:?}", now.elapsed());
        Ok(Self {
            interpreter: net,
            session,
        })
    }

    pub fn unload(self) -> Result<mnn::Interpreter> {
        let session = self.session;
        let net = self.interpreter;
        drop(session);
        Ok(net)
    }

    pub fn run_session(&mut self) -> Result<()> {
        self.interpreter.run_session(&self.session)
    }

    pub fn both_mut(&mut self) -> (&mut Interpreter, &mut Session) {
        (&mut self.interpreter, &mut self.session)
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

    fn run_callback(&mut self, f: Callback) -> Result<()> {
        #[cfg(feature = "tracing")]
        tracing::trace!("Running callback");
        #[cfg(feature = "tracing")]
        let now = std::time::Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(self)))
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
        #[cfg(feature = "tracing")]
        tracing::trace!("Callback took: {:?}", now.elapsed());
        result
    }
}

impl SessionHandle {
    pub fn new(interpreter: Interpreter, config: ScheduleConfig) -> Result<Self> {
        let (sender, receiver) = flume::unbounded::<CallbackSender>();
        let builder = std::thread::Builder::new().name("mnn-session-thread".to_string());
        let spawner = move || -> Result<()> {
            let mut ss = SessionState {
                sr: SessionRunnerState::Unloaded(interpreter),
                receiver,
                config,
            };

            loop {
                let cmd = ss
                    .receiver
                    .recv()
                    .change_context(ErrorKind::SyncError)
                    .attach_printable("Internal Error: Unable to recv (Sender possibly dropped without calling close)")?;
                match cmd {
                    CallbackEnum::Callback(f) => {
                        let sr = ss.sr().inspect_err(|e| {
                            #[cfg(feature = "tracing")]
                            tracing::error!("Error getting the session runtime :{:?}", e);
                        })?;
                        sr.run_callback(f)
                            .map_err(|e| e.into_inner())
                            .attach_printable("Failure running the callback")
                            .inspect_err(|e| {
                                #[cfg(feature = "tracing")]
                                tracing::error!("Error running callback: {:?}", e);
                            })?;
                    }
                    CallbackEnum::Unload(tx) => {
                        let res = ss.unload();
                        tx.send(res)
                            .change_context(ErrorKind::SyncError)
                            .attach_printable("Internal Error: Failed to send unload message")?;
                    }
                    CallbackEnum::Load(tx) => {
                        let res = ss.load();
                        tx.send(res)
                            .change_context(ErrorKind::SyncError)
                            .attach_printable("Internal Error: Failed to send load message")?;
                    }

                    CallbackEnum::Status(tx) => {
                        let res = ss.is_loaded();
                        tx.send(res)
                            .change_context(ErrorKind::SyncError)
                            .attach_printable("Internal Error: Failed to send status message")?;
                    }
                    CallbackEnum::Close => {
                        #[cfg(feature = "tracing")]
                        tracing::warn!("Closing session thread");
                        break;
                    }
                }
            }

            let SessionState {
                sr,
                receiver: _,
                config: _,
            } = ss;

            if let SessionRunnerState::Loaded(sr) = sr {
                #[cfg(feature = "tracing")]
                tracing::trace!("Unloading session before closing thread");
                sr.unload()
                    .change_context(ErrorKind::SyncError)
                    .attach_printable("Internal Error: Failed to unload session")?;
            } else if !sr.is_unloaded() {
                #[cfg(feature = "tracing")]
                tracing::warn!("Session was not loaded, no need to unload");
            }

            Ok(())
        };
        let handle = builder
            .spawn(spawner)
            .change_context(ErrorKind::SyncError)
            .attach_printable("Internal Error: Failed to spawn thread")?;

        Ok(Self {
            handle: Some(handle),
            sender,
        })
    }

    fn is_running(&self) -> bool {
        self.handle.as_ref().is_some_and(|j| !j.is_finished())
    }

    fn ensure_running(&self) -> Result<()> {
        if !self.is_running() {
            Err(Report::new(ErrorKind::SyncError).attach_printable("Session thread is not running"))?
        }
        Ok(())
    }

    pub fn run<R: Send + Sync + 'static>(
        &self,
        f: impl FnOnce(&mut SessionRunner) -> Result<R> + Send + Sync + 'static,
    ) -> Result<R> {
        self.ensure_running()?;
        #[cfg(feature = "tracing")]
        let span = tracing::Span::current();
        let f = f;
        let (tx, rx) = oneshot::channel();
        let wrapped_f = move |sr: &mut SessionRunner| -> Result<()> {
            #[cfg(feature = "tracing")]
            let _guard = span.enter();
            let result = f(sr);
            tx.send(result)
                .change_context(ErrorKind::SyncError)
                .attach_printable("Internal Error: Failed to send result via oneshot channel")?;
            Ok(())
        };
        self.sender
            .send(CallbackEnum::Callback(Box::new(wrapped_f)))
            .map_err(|e| Report::new(ErrorKind::SyncError).attach_printable(e.to_string()))?;
        rx.recv()
            .change_context(ErrorKind::SyncError)
            .attach_printable("Internal Error: Unable to recv message")?
    }

    pub async fn run_async<R: Send + Sync + 'static>(
        &self,
        f: impl FnOnce(&mut SessionRunner) -> Result<R> + Send + Sync + 'static,
    ) -> Result<R> {
        self.ensure_running()?;
        #[cfg(feature = "tracing")]
        let span = tracing::Span::current();
        let f = f;
        let (tx, rx) = oneshot::channel();
        let wrapped_f = move |sr: &mut SessionRunner| -> Result<()> {
            #[cfg(feature = "tracing")]
            let _guard = span.enter();
            let result = f(sr);
            tx.send(result)
                .change_context(ErrorKind::SyncError)
                .attach_printable("Internal Error: Failed to send result via oneshot channel")?;
            Ok(())
        };
        self.sender
            .send(CallbackEnum::Callback(Box::new(wrapped_f)))
            .map_err(|e| Report::new(ErrorKind::SyncError).attach_printable(e.to_string()))?;
        rx.await
            .change_context(ErrorKind::SyncError)
            .attach_printable("Internal Error: Unable to recv message")?
    }

    pub fn unload(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(CallbackEnum::Unload(tx))
            .map_err(|e| Report::new(ErrorKind::SyncError).attach_printable(e.to_string()))?;
        rx.recv()
            .change_context(ErrorKind::SyncError)
            .attach_printable("Internal Error: Failed to recv unload message")?
    }

    pub async fn unload_async(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(CallbackEnum::Unload(tx))
            .map_err(|e| Report::new(ErrorKind::SyncError).attach_printable(e.to_string()))?;
        rx.await
            .change_context(ErrorKind::SyncError)
            .attach_printable("Internal Error: Failed to recv unload message")?
    }

    pub fn load(&self) -> Result<()> {
        self.ensure_running()?;
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(CallbackEnum::Load(tx))
            .map_err(|e| Report::new(ErrorKind::SyncError).attach_printable(e.to_string()))?;
        rx.recv()
            .change_context(ErrorKind::SyncError)
            .attach_printable("Internal Error: Failed to recv load message")?
    }

    pub async fn load_async(&self) -> Result<()> {
        self.ensure_running()?;
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(CallbackEnum::Load(tx))
            .map_err(|e| Report::new(ErrorKind::SyncError).attach_printable(e.to_string()))?;
        rx.await
            .change_context(ErrorKind::SyncError)
            .attach_printable("Internal Error: Failed to recv load message")?
    }

    pub fn is_loaded(&self) -> Result<bool> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(CallbackEnum::Status(tx))
            .map_err(|e| Report::new(ErrorKind::SyncError).attach_printable(e.to_string()))?;
        Ok(rx
            .recv()
            .change_context(ErrorKind::SyncError)
            .attach_printable("Internal Error: Failed to recv status message")?)
    }

    pub fn close(&self) -> Result<()> {
        self.sender
            .send(CallbackEnum::Close)
            .map_err(|e| Report::new(ErrorKind::SyncError).attach_printable(e.to_string()))?;
        Ok(())
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
            input.copy_from_host_tensor(cpu_input.view())?;
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
    let interpreter = Interpreter::from_file("../tests/assets/realesr.mnn")
        .expect("Failed to create interpreter");
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
                    .copy_from_host_tensor(cpu_tensor.view())
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
                    .copy_from_host_tensor(cpu_tensor.view())
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
                    .copy_from_host_tensor(cpu_tensor.view())
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
        .expect("failed to copy output");
}

#[test]
pub fn test_sync_api_is_send_sync() {
    fn is_send_sync<T: Send + Sync>() {}
    is_send_sync::<SessionHandle>();
}

#[test]
#[cfg_attr(feature = "tracing", tracing_test::traced_test)]
pub fn test_load_unload() {
    let interpreter = Interpreter::from_file("../tests/assets/realesr.mnn")
        .expect("Failed to create interpreter");
    let session_handle = SessionHandle::new(interpreter, ScheduleConfig::new())
        .expect("Failed to create session handle");
    session_handle
        .run(|sr| {
            for input in sr.interpreter.inputs(sr.session()).iter() {
                input
                    .tensor::<f32>()
                    .expect("Failed to get tensor")
                    .fill(1.0f32);
            }
            Ok(())
        })
        .expect("Failed to run");
    session_handle.load().expect("Failed to load");
    session_handle.unload().expect("Failed to unload");
    session_handle.load().expect("Failed to load");
    session_handle.unload().expect("Failed to unload");
    session_handle
        .run(|sr| {
            sr.run_session()?;
            Ok(())
        })
        .expect("Failed to run");
    session_handle.unload().expect("Failed to unload");
}
