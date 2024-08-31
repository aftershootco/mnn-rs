use crate::prelude::*;
use crate::*;

type Callback = Box<dyn FnOnce(&mut SessionRunner) -> Result<()> + Send + Sync + 'static>;
pub enum CallbackEnum {
    Callback(Callback),
    Close,
}
type CallbackSender = (CallbackEnum, oneshot::Sender<Result<()>>);

pub struct SessionHandle {
    #[allow(dead_code)]
    pub(crate) handle: std::thread::JoinHandle<Result<()>>,
    pub(crate) sender: std::sync::mpsc::Sender<CallbackSender>,
}

impl Drop for SessionHandle {
    fn drop(&mut self) {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send((CallbackEnum::Close, tx))
            .expect("Failed to close SessionHandle");
        // rx.recv().expect("Failed to close SessionHandle");
    }
}

pub struct SessionRunner {
    pub(crate) interpreter: Interpreter,
    pub(crate) session: Session,
}

impl SessionHandle {
    pub fn new(mut interpreter: Interpreter, mut config: ScheduleConfig) -> Result<Self> {
        let (sender, receiver) = std::sync::mpsc::channel::<CallbackSender>();
        let handle = std::thread::spawn(move || -> Result<()> {
            let session = interpreter.create_session(&mut config)?;
            let mut session_runner = SessionRunner {
                interpreter,
                session,
            };
            loop {
                let (f, tx) = receiver
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
                    Err(Report::new(ErrorKind::SyncError).attach_printable(format!("{:?}", e)))
                });
                tx.send(result)
                    .change_context(ErrorKind::SyncError)
                    .attach_printable(
                        "Internal Error: Failed to send result via oneshot channel",
                    )?;
            }
            Ok(())
        });
        Ok(Self { handle, sender })
    }

    pub fn run(
        &mut self,
        f: impl FnOnce(&mut SessionRunner) -> Result<()> + Send + Sync + 'static,
    ) -> Result<()> {
        let f = Box::new(f);
        let (tx, rx) = oneshot::channel();
        self.sender
            .send((CallbackEnum::Callback(f), tx))
            .map_err(|e| Report::new(ErrorKind::SyncError).attach_printable(e.to_string()))?;
        rx.recv()
            .change_context(ErrorKind::SyncError)
            .attach_printable("Internal Error: Unable to recv message")?
            .attach_printable("Callback Error: Error in the provided callback")?;
        Ok(())
    }
}

impl SessionRunner {
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
    let mut session_handle =
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
            sr.interpreter().run_session(&sr.session())?;
            Ok(())
        })
        .expect("Failed to run");

    session_handle
        .run(|sr| {
            let output = sr.interpreter().output::<f32>(&sr.session(), "output")?;
            let cpu_output = output.create_host_tensor_from_device(true);
            cpu_output.host().to_vec();
            Ok(())
        })
        .expect("Sed");
}

#[test]
pub fn test_sync_api_race() {
    let interpreter =
        Interpreter::from_file("./aot_all.mnn").expect("Failed to create interpreter");
    let mut session_handle = SessionHandle::new(interpreter, ScheduleConfig::new())
        .expect("Failed to create session handle");
    session_handle
        .run(move |sr| {
            let session = sr.session();
            let interpreter = sr.interpreter();
            let inputs = interpreter.inputs(&session);
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
            sr.interpreter().run_session(&sr.session())?;
            Ok(())
        })
        .expect("Failed to run");
    session_handle
        .run(move |sr| {
            let session = sr.session();
            let interpreter = sr.interpreter();
            let inputs = interpreter.inputs(&session);
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
            let inputs = interpreter.inputs(&session);
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
            sr.interpreter().run_session(&sr.session())?;
            Ok(())
        })
        .expect("Failed to run");

    session_handle
        .run(|sr| {
            let output = sr.interpreter().output::<f32>(&sr.session(), "output")?;
            let cpu_output = output.create_host_tensor_from_device(true);
            cpu_output.host().to_vec();
            Ok(())
        })
        .expect("Sed");
}
