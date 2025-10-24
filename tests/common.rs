pub use mnn::*;
pub type Error = Box<dyn std::error::Error + Send + Sync + 'static>;
pub type Result<T, E = Error> = std::result::Result<T, E>;
pub struct Model {
    bytes: &'static [u8],
}

impl Model {
    pub const fn new() -> Self {
        Model {
            bytes: include_bytes!("assets/realesr.mnn"),
        }
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

impl AsRef<[u8]> for Model {
    fn as_ref(&self) -> &[u8] {
        self.bytes
    }
}

#[allow(dead_code)]
pub fn test_basic(backend: ForwardType) -> Result<()> {
    let mut net = mnn::Interpreter::from_file("tests/assets/realesr.mnn")?;
    let mut config = ScheduleConfig::new();
    config.set_type(backend);
    let session = net.create_session(config)?;
    net.inputs(&session).iter_mut().for_each(|x| {
        let mut tensor = x.tensor::<f32>().expect("No tensor");
        println!("{}: {:?}", x.name(), tensor.shape());
        tensor.fill(1.0f32);
    });
    net.run_session(&session)?;
    let outputs = net.outputs(&session);
    for output in outputs.iter() {
        println!("output: {:?}", output);
        let tensor = output.tensor::<f32>()?;
        let shape = tensor.shape();
        assert_eq!(shape.as_ref(), [1, 3, 2048, 2048]);
    }
    Ok(())
}

#[allow(dead_code)]
pub fn test_multipath_session(backend: ForwardType, backend2: ForwardType) -> Result<()> {
    use mnn::BackendConfig;

    let mut net = mnn::Interpreter::from_bytes(Model::new())?;
    let mut config = ScheduleConfig::new();
    config.set_type(backend);
    config.set_backup_type(backend);
    let mut bc = BackendConfig::new();
    bc.set_memory_mode(mnn::MemoryMode::High);
    bc.set_precision_mode(mnn::PrecisionMode::High);
    bc.set_power_mode(mnn::PowerMode::High);
    let mut config2 = ScheduleConfig::new();
    config2.set_type(backend2);
    config2.set_backup_type(backend2);
    let mut bc = BackendConfig::new();
    bc.set_memory_mode(mnn::MemoryMode::High);
    bc.set_precision_mode(mnn::PrecisionMode::High);
    bc.set_power_mode(mnn::PowerMode::High);
    config2.set_backend_config(bc);

    let session = net.create_multipath_session([config, config2])?;
    {
        let inputs = net.inputs(&session);
        for input in inputs.iter() {
            println!("input: {:?}", input);
            input.tensor::<f32>()?.fill(0.0);
        }
    }
    net.run_session(&session)?;
    let outputs = net.outputs(&session);
    for output in outputs.iter() {
        println!("output: {:?}", output);
        let tensor = output.tensor::<f32>()?;
        let shape = tensor.shape();
        assert_eq!(shape.as_ref(), [1, 3, 2048, 2048]);
    }
    Ok(())
}
