use burn::backend::{Autodiff, Wgpu};

use crate::training::run;

mod data;
mod flow;
mod model;
mod training;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = burn::backend::wgpu::WgpuDevice::default();

    run::<MyAutodiffBackend>(&device);
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use crate::model::UNet;
    use crate::model::time_embedding;

    use super::*;
    use burn::tensor::Float;
    use burn::tensor::Tensor;

    fn parse_pytorch_tensor_output(output: &str) -> Vec<Vec<f32>> {
        // Remove prefix/suffix like "tensor(" and ")"
        let clean = output
            .replace("tensor(", "")
            .replace("tensor", "")
            .replace("(", "")
            .replace(")", "")
            .replace("[[", "")
            .replace("]]", "")
            .replace("]", "")
            .replace("[", "")
            .replace("\n", " ")
            .replace(",", " ");

        // Split into tokens and parse numbers
        let nums: Vec<f32> = clean
            .split_whitespace()
            .filter_map(|s| s.parse::<f32>().ok())
            .collect();

        // Infer number of columns by counting commas in the first row
        // or, better, pass known output_dim from context
        let output_dim = 8; // you know this from the embedding setup
        let mut rows = Vec::new();
        for chunk in nums.chunks(output_dim) {
            rows.push(chunk.to_vec());
        }

        rows
    }

    #[test]
    pub fn test_time_embedding() {
        let ts = [0.1, 0.2, 0.3];
        let output_dim = 8;
        let py_code = format!(
            r#"
import torch

def time_embedding(ts, output_dim):
    i = torch.arange(0, output_dim)
    log_term = -torch.log(torch.tensor(10000.0)) / output_dim
    div_term = torch.exp(i * log_term)
    angle = ts[:, None] * div_term[None, :]
    v = torch.zeros(ts.size(0), output_dim)
    v[:, 0::2] = torch.sin(angle[:, 0::2])
    v[:, 1::2] = torch.cos(angle[:, 1::2])
    return v

ts = torch.tensor({timesteps})
emb = time_embedding(ts, {output_dim})
torch.set_printoptions(precision=10, sci_mode=False)
print(emb)
"#,
            timesteps = format!("{:?}", ts),
            output_dim = output_dim,
        );
        let py_output = Command::new("python")
            .arg("-c")
            .arg(py_code)
            .output()
            .expect("failed to run Python");

        let py_stdout = String::from_utf8_lossy(&py_output.stdout);
        let py_tensor = parse_pytorch_tensor_output(&py_stdout);

        type MyBackend = Wgpu<f32, i32>;
        let device = Default::default();
        let embedding = time_embedding::<MyBackend>(
            &Tensor::<MyBackend, 1, Float>::from_floats(ts, &device),
            8,
            &device,
        );
        let rs_tensor: Vec<f32> = embedding.into_data().into_vec().unwrap();

        for (r, p) in rs_tensor.iter().zip(py_tensor.iter().flatten()) {
            if (r - p).abs() >= 1e-5 {
                println!("{:?}", py_tensor);
                println!("{:?}", rs_tensor);
                panic!("Embedding Mismatch: Burn {r} vs PyTorch {p}")
            }
        }
    }

    #[test]
    pub fn test_unet() {
        type MyBackend = Wgpu<f32, i32>;
        let device = Default::default();
        let model = UNet::init(3, 128, 10, &device);

        let x = Tensor::<MyBackend, 4>::zeros([2, 3, 64, 64], &device); // [batch, in_ch, H, W]
        let time_emb = time_embedding(
            &Tensor::<MyBackend, 1>::from_floats([0.3, 0.6], &device),
            128,
            &device,
        ); // [batch, time_dim]
        let cond_emb = Tensor::<MyBackend, 2>::zeros([2, 64], &device); // [batch, cond_dim]

        let y = model.forward(x, time_emb, cond_emb);
        assert!(y.dims() == [2, 3, 64, 64]);
    }
}
