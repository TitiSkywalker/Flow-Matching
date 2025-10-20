# Flow Matching for Image Generation

![alt text](./plot/sample.png)

This repository implements an image generation model in Rust using the Burn framework. This model is a flow matching model trained on MNIST dataset. 

To run the training and inference code: 

```
cargo run --release
```

The model will be saved in the "./model" directory, and the generated samples will be saved in the "./result" directory.

These are the default training configurations. You might need to adjust the batch size to match your own device.

```Rust
#[derive(Config)]
pub struct TrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 1)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1e-4)]
    pub lr: f64,
}
```