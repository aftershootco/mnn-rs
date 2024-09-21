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
            if dbg!(session.has_async_work()) {
                dbg!("has async work");
                return Poll::Pending;
            }
            net.outputs(session).iter().all(|output| {
                dbg!(output
                    .raw_tensor()
                    .wait(ffi::MapType::MAP_TENSOR_READ, true))
            });
            Poll::Ready(Ok(()))
        } else {
            self.running
                .store(true, std::sync::atomic::Ordering::SeqCst);
            let (net, session) = &mut self.inner;
            let now = std::time::Instant::now();
            net.run_session(&session)?;
            dbg!(now.elapsed());
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
    }
}

#[test]
fn test_async_run_session() {
    let mut schedule_config = ScheduleConfig::new();
    schedule_config.set_type(ForwardType::OpenCL);
    schedule_config.set_backup_type(ForwardType::OpenCL);
    let mut interpreter = Interpreter::from_file("skin_retouching.mnn").unwrap();
    interpreter
        .set_cache_file("skin_retouching.cache", 128)
        .unwrap();
    interpreter.set_session_mode(SessionMode::Release);
    let mut session = interpreter.create_session(schedule_config).unwrap();
    interpreter.update_cache_file(&mut session).unwrap();
    dbg!("loaded");
    interpreter
        .input::<f32>(&session, "image")
        .unwrap()
        .fill(1.0f32);
    let now = std::time::Instant::now();
    smol::block_on(async {
        interpreter.run_session_(&session).await.unwrap();
    });

    // interpreter
    //     .run_session_with_callback_info(&session, TensorCallback::identity(), |_, op| true, true)
    // .unwrap();
    interpreter.outputs(&session).iter().for_each(|output| {
        dbg!(output
            .raw_tensor()
            .wait(mnn::ffi::MapType::MAP_TENSOR_READ, false));
    });
    dbg!("end");
    println!("time: {:?}", now.elapsed());
}
