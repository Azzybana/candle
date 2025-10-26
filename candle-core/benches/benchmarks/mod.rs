pub(crate) mod affine;
pub(crate) mod conv_transpose2d;
pub(crate) mod copy;
pub(crate) mod matmul;
pub(crate) mod qmatmul;
pub(crate) mod random;
pub(crate) mod reduce;
pub(crate) mod unary;
pub(crate) mod where_cond;

use candle_core::{Device, Result};

pub(crate) trait BenchDevice {
    fn sync(&self) -> Result<()>;

    fn bench_name<S: Into<String>>(&self, name: S) -> String;
}

impl BenchDevice for Device {
    fn sync(&self) -> Result<()> {
        match self {
            Device::Cpu => Ok(()),
        }
    }

    fn bench_name<S: Into<String>>(&self, name: S) -> String {
        match self {
            Device::Cpu => {
                let cpu_type = if cfg!(feature = "mkl") { "mkl" } else { "cpu" };
                format!("{}_{}", cpu_type, name.into())
            }
        }
    }
}

struct BenchDeviceHandler {
    devices: Vec<Device>,
}

impl BenchDeviceHandler {
    pub fn new() -> Result<Self> {
        let devices = vec![Device::Cpu];
        Ok(Self { devices })
    }
}
