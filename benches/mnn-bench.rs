use divan::*;
#[divan::bench_group(sample_size = 5, sample_count = 5)]
mod mnn_realesr_bench_with_ones {
    use divan::*;
    use mnn::*;
    #[divan::bench]
    pub fn mnn_realesr_benchmark_cpu(bencher: Bencher) {
        let mut net = Interpreter::from_file("tests/assets/realesr.mnn").unwrap();
        let mut config = ScheduleConfig::new();
        config.set_type(ForwardType::CPU);
        let session = net.create_session(config).unwrap();
        bencher.bench_local(|| {
            let mut input = net.input(&session, "data").unwrap();
            input.fill(1f32);
            net.run_session(&session).unwrap();
        });
    }

    #[cfg(feature = "opencl")]
    #[divan::bench]
    pub fn mnn_realesr_benchmark_opencl(bencher: Bencher) {
        let mut net = Interpreter::from_file("tests/assets/realesr.mnn").unwrap();
        let mut config = ScheduleConfig::new();
        config.set_type(ForwardType::OpenCL);
        let session = net.create_session(config).unwrap();
        bencher.bench_local(|| {
            let mut input = net.input(&session, "data").unwrap();
            input.fill(1f32);
            net.run_session(&session).unwrap();
            net.wait(&session);
        });
    }

    #[cfg(feature = "metal")]
    #[divan::bench]
    pub fn mnn_realesr_benchmark_metal(bencher: Bencher) {
        let mut net = Interpreter::from_file("tests/assets/realesr.mnn").unwrap();
        let mut config = ScheduleConfig::new();
        config.set_type(ForwardType::Metal);
        let session = net.create_session(config).unwrap();
        bencher.bench_local(|| {
            let mut input = net.input(&session, "data").unwrap();
            input.fill(1f32);
            net.run_session(&session).unwrap();
            net.wait(&session);
        });
    }

    #[cfg(feature = "coreml")]
    #[divan::bench]
    pub fn mnn_realesr_benchmark_coreml(bencher: Bencher) {
        let mut net = Interpreter::from_file("tests/assets/realesr.mnn").unwrap();
        let mut config = ScheduleConfig::new();
        config.set_type(ForwardType::CoreML);
        let session = net.create_session(config).unwrap();
        bencher.bench_local(|| {
            let mut input = net.input(&session, "data").unwrap();
            input.fill(1f32);
            net.run_session(&session).unwrap();
            net.wait(&session);
        });
    }
}
