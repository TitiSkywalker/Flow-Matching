#![recursion_limit = "256"]
use burn::{
    backend::{Autodiff, Wgpu},
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
};

use crate::{
    inference::{guided_sampling_flow, save_tensor_image},
    model::UNet,
    training::{TrainingConfig, train},
};

mod data;
mod flow;
mod inference;
mod model;
mod training;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = burn::backend::wgpu::WgpuDevice::default();

    let model = train::<MyAutodiffBackend>(TrainingConfig::new(), &device);

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .save_file("./model/flow.mpk", &recorder)
        .expect("Failed to save model");

    let model = UNet::<MyBackend>::init(1, 128, 10, &device)
        .load_file("./model/flow.mpk", &recorder, &device)
        .expect("Failed to load model");

    for number in 0..10 {
        let image = guided_sampling_flow(&model, number, 100, &device);
        save_tensor_image(
            &image,
            &format!("./result/sample_{}.png", number).to_string(),
        );
    }
}
