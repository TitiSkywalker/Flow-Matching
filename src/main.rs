#![recursion_limit = "256"]
use std::path::Path;

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
    let model_file = "./model/flow.mpk";
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    let mut model = UNet::<MyAutodiffBackend>::init(1, 128, 10, &device);
    if Path::new(model_file).exists() {
        println!("Resume from existing model");
        model = UNet::<MyAutodiffBackend>::init(1, 128, 10, &device)
            .load_file(model_file, &recorder, &device)
            .expect("Failed to load model");
    }

    let trained_model = train::<MyAutodiffBackend>(TrainingConfig::new(), model, &device);

    println!("Save trained model");
    trained_model
        .save_file(model_file, &recorder)
        .expect("Failed to save model");

    println!("Reload model for inference");
    let infer_model = UNet::<MyBackend>::init(1, 128, 10, &device)
        .load_file(model_file, &recorder, &device)
        .expect("Failed to load model");

    println!("Generate samples");
    for number in 0..10 {
        println!("{number}");
        let image = guided_sampling_flow(&infer_model, number, 100, &device);
        save_tensor_image(
            &image,
            &format!("./result/sample_{}.png", number).to_string(),
            4,
        );
    }
    println!("Finish");
}
