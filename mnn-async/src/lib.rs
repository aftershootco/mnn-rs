use std::sync::{atomic::AtomicBool, Arc};

use mnn::*;
pub trait AsyncInterpreter {
    fn run_session_<'r>(
        &'r mut self,
        session: &'r crate::session::Session,
    ) -> impl core::future::Future<Output = Result<()>> + 'r;
}

impl AsyncInterpreter for Interpreter {
    fn run_session_<'r>(
        &'r mut self,
        session: &'r crate::session::Session,
    ) -> impl core::future::Future<Output = Result<()>> + 'r {
        RunSession {
            inner: (self, session),
            running: Arc::new(AtomicBool::new(false)),
        }
    }
}

pub struct RunSession<'r, 's> {
    pub(crate) inner: (&'r mut Interpreter, &'s crate::session::Session),
    pub(crate) running: Arc<AtomicBool>,
}

impl<'r, 's> core::future::Future for RunSession<'r, 's> {
    type Output = Result<()>;
    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        use std::task::Poll;

        if self.running.load(std::sync::atomic::Ordering::SeqCst) {
            let (net, session) = &mut self.inner;
            println!("polling...");
            if net.outputs(session).iter().all(|output| {
                dbg!(output
                    .raw_tensor()
                    .wait(ffi::MapType::MAP_TENSOR_READ, true))
            }) {
                println!("done!");
                Poll::Ready(Ok(()))
            } else {
                println!("pending...");
                Poll::Pending
            }
        } else {
            self.running
                .store(true, std::sync::atomic::Ordering::SeqCst);
            let (net, session) = &mut self.inner;
            let now = std::time::Instant::now();
            let waker = cx.waker().clone();
            net.run_session_with_callback(
                session,
                |_, _| 1,
                move |_, _| {
                    waker.wake_by_ref();
                    1
                },
                false,
            )?;
            println!("timing: {:?}", now.elapsed());
            return Poll::Pending;
        }
    }
}

#[test]
fn test_async_run_session() {
    let mut schedule_config = ScheduleConfig::new();
    schedule_config.set_type(ForwardType::OpenCL);
    let mut interpreter = Interpreter::from_file("../tests/assets/realesr.mnn").unwrap();
    interpreter
        .set_cache_file("../tests/assets/realesr.cache", 128)
        .unwrap();
    let mut session = interpreter.create_session(schedule_config).unwrap();
    interpreter.update_cache_file(&mut session).unwrap();
    interpreter
        .input::<f32>(&session, "data")
        .unwrap()
        .fill(1.0f32);
    let now = std::time::Instant::now();
    interpreter
        .run_session_with_callback_info(
            &session,
            |_, _| 1,
            |_, op| {
                std::thread::sleep(core::time::Duration::from_secs(5));
                dbg!(op);
                1
            },
            false,
        )
        .unwrap();
    println!("time: {:?}", now.elapsed());
}
