use std::time::Instant;

use crate::model::time_embedding;
use crate::{data::FlowBatcher, model::UNet};
use burn::data::dataset::Dataset;
use burn::nn::loss::Reduction;
use burn::optim::GradientsParams;
use burn::optim::Optimizer;
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    nn::loss::MseLoss,
    optim::AdamConfig,
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Config)]
pub struct TrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 4)]
    pub batch_size: usize,
    #[config(default = 1)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub lr: f64,
}

pub fn run<B: AutodiffBackend>(device: &B::Device) {
    // Create the configuration
    println!("Create configurations");
    let config = TrainingConfig::new();

    B::seed(config.seed);

    // Create model and optimizer
    println!("Create model and optimizer");
    let mut model = UNet::<B>::init(1, 128, 10, &device);
    let mut optim = AdamConfig::new().init();

    // Create the batcher
    println!("Create batcher");
    let batcher = FlowBatcher::default();

    // Create dataloaders
    println!("Create dataloader");
    let dataloader = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());
    let total_batches = MnistDataset::train().len() / config.batch_size;

    // Iterate over training loop
    for epoch in 1..config.num_epochs + 1 {
        // Create a tqdm-like progress bar
        let pb = ProgressBar::new(total_batches as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{elapsed_precise} | ETA {eta_precise} [{bar:40.cyan/blue}] {pos}/{len} | loss={msg}",
            )
            .unwrap()
            .progress_chars("=> "),
        );
        let mut running_loss = 0.0f64;
        let start_time = Instant::now();

        // Implement the training loop
        for (i, batch) in dataloader.iter().enumerate() {
            let time_embed = time_embedding(&batch.t.clone(), 128, device);

            let u = model.forward(batch.x, time_embed, batch.cond);

            let loss = MseLoss::new().forward(u, batch.u_target, Reduction::Auto);

            let loss_value: f64 = loss.clone().into_scalar().elem();
            running_loss += loss_value;

            // Gradients for the current backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            // Update the model using the optimizer
            model = optim.step(config.lr, model, grads);

            // Free graph memory
            loss.detach();

            // Update prgress bar
            let avg_loss = running_loss / (i as f64 + 1.0);
            pb.set_message(format!("{:.4}", avg_loss));
            pb.inc(1);
        }

        pb.finish_with_message(format!(
            "epoch {} finished | avg_loss = {:.4} | time = {:.2?}",
            epoch,
            running_loss / total_batches as f64,
            start_time.elapsed()
        ));
    }
}
