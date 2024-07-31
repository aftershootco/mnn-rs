use mnn_sys::{MNNForwardType, PowerMode, PrecisionMode};

#[derive(clap::ValueEnum, Debug, Clone, Default)]
pub enum ForwardType {
    All,
    #[default]
    Auto,
    CPU,
    Metal,
    OpenCL,
    OpenGL,
    Vulkan,
    CoreML,
}

impl ForwardType {
    pub fn to_forward_type(&self) -> MNNForwardType {
        match self {
            ForwardType::Auto => MNNForwardType::MNN_FORWARD_AUTO,
            ForwardType::All => MNNForwardType::MNN_FORWARD_ALL,
            ForwardType::CPU => MNNForwardType::MNN_FORWARD_CPU,
            ForwardType::Metal => MNNForwardType::MNN_FORWARD_METAL,
            ForwardType::OpenCL => MNNForwardType::MNN_FORWARD_OPENCL,
            ForwardType::OpenGL => MNNForwardType::MNN_FORWARD_OPENGL,
            ForwardType::Vulkan => MNNForwardType::MNN_FORWARD_VULKAN,
            ForwardType::CoreML => MNNForwardType::MNN_FORWARD_NN,
        }
    }
}

#[derive(clap::ValueEnum, Debug, Clone)]
pub enum Modes {
    Low,
    Normal,
    High,
}

impl Modes {
    pub fn to_precision_mode(&self) -> PrecisionMode {
        match self {
            Modes::Low => PrecisionMode::Precision_Low,
            Modes::Normal => PrecisionMode::Precision_Normal,
            Modes::High => PrecisionMode::Precision_High,
        }
    }
    pub fn to_power_mode(&self) -> PowerMode {
        match self {
            Modes::Low => PowerMode::Power_Low,
            Modes::Normal => PowerMode::Power_Normal,
            Modes::High => PowerMode::Power_High,
        }
    }
}
