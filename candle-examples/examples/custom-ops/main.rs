// This example illustrates how to implement custom operations. These operations can provide their
// own forward pass (CPU and GPU versions) as well as their backward pass.
//
// In this example we add the RMS normalization operation and implement it for f32.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use clap::Parser;

use candle::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
}

struct LayerNorm {
    eps: f32,
}

impl CustomOp1 for LayerNorm {
    fn name(&self) -> &'static str {
        "layer-norm"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let (dim1, dim2) = layout.shape().dims2()?;
        let slice = storage.as_slice::<f32>()?;
        let src = match layout.contiguous_offsets() {
            None => candle::bail!("input has to be contiguous"),
            Some((o1, o2)) => &slice[o1..o2],
        };
        let mut dst = Vec::with_capacity(dim1 * dim2);
        for idx1 in 0..dim1 {
            let src = &src[idx1 * dim2..(idx1 + 1) * dim2];
            let variance = src.iter().map(|x| x * x).sum::<f32>();
            let s_variance = 1f32 / (variance / dim2 as f32 + self.eps).sqrt();
            dst.extend(src.iter().map(|x| x * s_variance))
        }
        let storage = candle::WithDType::to_cpu_storage_owned(dst);
        Ok((storage, layout.shape().clone()))
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let t = Tensor::arange(0f32, 14f32, &device)?.reshape((2, 7))?;
    println!("{t}");
    let t = t.apply_op1(LayerNorm { eps: 1e-5 })?;
    println!("{t}");
    Ok(())
}
