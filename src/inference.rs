use std::fs;
use std::path::Path;

use crate::model::UNet;
use crate::model::time_embedding;
use burn::tensor::Float;
use burn::tensor::{Tensor, backend::Backend};
use image::GrayImage;
use image::Luma;
use image::imageops::FilterType;
use image::{Rgb, RgbImage};

pub fn guided_sampling_flow<B: Backend>(
    model: &UNet<B>,
    prompt: usize,
    num_steps: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let step_size = 1.0 / num_steps as f32;
    let mut t: f32 = 0.0;
    let mut x = Tensor::<B, 4>::random(
        [1, 1, 28, 28],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );

    let prompt = Tensor::<B, 1, Float>::from_floats([prompt], device).one_hot(10);

    for _ in 0..num_steps {
        let time_embed = time_embedding(&Tensor::<B, 1>::from_floats([t], device), 128, device);

        // Predict vector field
        let u = model.forward(x.clone(), time_embed.clone(), prompt.clone());

        x = x + u * step_size;
        t += step_size;
    }
    x.squeeze_dim(0)
}

pub fn save_tensor_image<B: Backend>(tensor: &Tensor<B, 3>, path: &str, scale: u32) {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent).expect("Failed to create directory");
    }

    let data = tensor.clone().to_data();
    let values: Vec<f32> = data.clone().convert::<f32>().into_vec().unwrap();
    let shape = data.shape;

    assert!(shape.len() == 3, "Expected tensor of shape [C,H,W]");
    let (c, h, w) = (shape[0], shape[1], shape[2]);
    let clipped: Vec<f32> = values.into_iter().map(|v| v.clamp(0.0, 1.0)).collect();

    match c {
        1 => {
            let mut img = GrayImage::new(w as u32, h as u32);
            for y in 0..h {
                for x in 0..w {
                    let idx = y * w + x;
                    let val = (clipped[idx] * 255.0) as u8;
                    img.put_pixel(x as u32, y as u32, Luma([val]));
                }
            }
            let upscaled = image::imageops::resize(
                &img,
                w as u32 * scale,
                h as u32 * scale,
                FilterType::Nearest,
            );
            upscaled.save(path).expect("Failed to save grayscale image");
        }
        3 => {
            let mut img = RgbImage::new(w as u32, h as u32);
            for y in 0..h {
                for x in 0..w {
                    let idx = y * w + x;
                    let r = (clipped[idx] * 255.0) as u8;
                    let g = (clipped[idx + w * h] * 255.0) as u8;
                    let b = (clipped[idx + 2 * w * h] * 255.0) as u8;
                    img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
                }
            }
            let upscaled = image::imageops::resize(
                &img,
                w as u32 * scale,
                h as u32 * scale,
                FilterType::Nearest,
            );
            upscaled.save(path).expect("Failed to save RGB image");
        }
        _ => panic!("Only channel = 1 or 3 supported, got channel = {}", c),
    }
}
