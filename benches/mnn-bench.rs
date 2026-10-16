use divan::*;
#[divan::bench_group(sample_size = 5, sample_count = 5)]
mod mnn_realesr_bench_with_ones {
    use divan::*;
    use mnn::*;
    #[divan::bench]
    pub fn mnn_realesr_benchmark_cpu(bencher: Bencher) {
        let net = Interpreter::from_file("tests/assets/realesr.mnn").unwrap();
        let mut config = ScheduleConfig::new();
        config.set_type(ForwardType::CPU);
        let mut session = net.create_session(config).unwrap();
        bencher.bench_local(|| {
            let input = net.input(&mut session, "data").unwrap();
            input.fill(1f32).unwrap();
            net.run_session(&session).unwrap();
        });
    }

    #[cfg(feature = "opencl")]
    #[divan::bench]
    pub fn mnn_realesr_benchmark_opencl(bencher: Bencher) {
        let net = Interpreter::from_file("tests/assets/realesr.mnn").unwrap();
        let mut config = ScheduleConfig::new();
        config.set_type(ForwardType::OpenCL);
        let mut session = net.create_session(config).unwrap();
        bencher.bench_local(|| {
            let input = net.input(&mut session, "data").unwrap();
            input.fill(1f32).unwrap();
            net.run_session(&session).unwrap();
            net.wait(&session);
        });
    }

    #[cfg(feature = "metal")]
    #[divan::bench]
    pub fn mnn_realesr_benchmark_metal(bencher: Bencher) {
        let net = Interpreter::from_file("tests/assets/realesr.mnn").unwrap();
        let mut config = ScheduleConfig::new();
        config.set_type(ForwardType::Metal);
        let mut session = net.create_session(config).unwrap();
        bencher.bench_local(|| {
            let input = net.input(&mut session, "data").unwrap();
            input.fill(1f32).unwrap();
            net.run_session(&session).unwrap();
            net.wait(&session);
        });
    }

    #[cfg(feature = "cuda")]
    #[divan::bench]
    pub fn mnn_realesr_benchmark_cuda(bencher: Bencher) {
        let net = Interpreter::from_file("tests/assets/realesr.mnn").unwrap();
        let mut config = ScheduleConfig::new();
        config.set_type(ForwardType::Cuda);
        let mut session = net.create_session(config).unwrap();
        bencher.bench_local(|| {
            let input = net.input(&mut session, "data").unwrap();
            input.fill(1f32).unwrap();
            net.run_session(&session).unwrap();
            net.wait(&session);
        });
    }

    #[cfg(feature = "coreml")]
    #[divan::bench]
    pub fn mnn_realesr_benchmark_coreml(bencher: Bencher) {
        let net = Interpreter::from_file("tests/assets/realesr.mnn").unwrap();
        let mut config = ScheduleConfig::new();
        config.set_type(ForwardType::CoreML);
        let mut session = net.create_session(config).unwrap();
        bencher.bench_local(|| {
            let input = net.input(&mut session, "data").unwrap();
            input.fill(1f32).unwrap();
            net.run_session(&session).unwrap();
            net.wait(&session);
        });
    }
}
