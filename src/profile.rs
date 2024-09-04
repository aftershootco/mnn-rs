#[cfg(feature = "profile")]
macro_rules! profile {
    ($message: expr; $($t:tt)*) => {{
        let now = std::time::Instant::now();
        #[cfg(feature = "tracing")]
        tracing::trace!("{}: Starting", $message);
        let result = {
            $($t)*
        };
        let elapsed = now.elapsed();
        #[cfg(feature = "tracing")]
        tracing::trace!("{}: Elapsed time: {:?}", $message, elapsed);
        result
    }}
}
#[cfg(not(feature = "profile"))]
macro_rules! profile {
    ($_: expr; $($t:tt)*) => {
        $($t)*
    }
}
pub(crate) use profile;

#[test]
pub fn test_profiling() {
    let time = std::time::Instant::now();
    profile!("Testing profiling"; {
        std::thread::sleep(std::time::Duration::from_secs(1));
    });
    let time = time.elapsed();
    assert!(time.as_secs() == 1);
}
