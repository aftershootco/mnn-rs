mod basic;
mod common;
use common::Result;
use mnn::ForwardType;

#[cfg(test)]
pub fn test_multi_threading(backend: ForwardType) -> Result<()> {
    let handles: Vec<_> = (1..=10)
        .map(move |_| std::thread::spawn(move || basic::test_basic(backend)))
        .collect();
    handles
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect::<Result<Vec<_>>>()?;
    Ok(())
}

#[test]
#[ignore = "takes too long"]
fn test_multi_threading_cpu() {
    test_multi_threading(ForwardType::CPU).unwrap();
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "takes too long"]
fn test_multi_threading_metal() {
    test_multi_threading(ForwardType::Metal).unwrap();
}

#[cfg(feature = "opencl")]
#[test]
#[ignore = "takes too long"]
fn test_multi_threading_opencl() {
    test_multi_threading(ForwardType::OpenCL).unwrap();
}
