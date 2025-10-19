use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
    tensor::Distribution,
};

use crate::flow::gaussian_vector_field;

#[derive(Clone, Default)]
pub struct FlowBatcher {}

#[derive(Clone, Debug)]
pub struct FlowBatch<B: Backend> {
    pub x: Tensor<B, 4>,        // (B, 1, 28, 28)
    pub t: Tensor<B, 1>,        // (B,)
    pub cond: Tensor<B, 2>,     // (B, 10)
    pub u_target: Tensor<B, 4>, // (B, 1, 28, 28)
}

impl<B: Backend> Batcher<B, MnistItem, FlowBatch<B>> for FlowBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> FlowBatch<B> {
        // Load and normalize MNIST images
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image.clone()).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 2>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            .map(|tensor| tensor / 255)
            .collect::<Vec<_>>();

        // println!("{}", images[0]);

        let z = Tensor::stack(images, 0); // (B, 1, 28, 28)
        let batch_size = z.dims()[0];

        // Prepare label tensor
        let labels = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data([(item.label as i64).elem::<B::IntElem>()], device)
            })
            .collect::<Vec<_>>();
        let labels = Tensor::cat(labels, 0); // (B,)

        // Sample random timesteps t in [0,1] (B,)
        let t: Tensor<B, 1> =
            Tensor::<B, 1>::random([batch_size], Distribution::Uniform(0.0, 1.0), &device);

        // Build one-hot conditional embedding (B, 10)
        let cond: Tensor<B, 2> = labels.float().one_hot(10);

        // Compute flow-matching trajectory (x, u_target)
        let (x, u_target) = gaussian_vector_field(&t, &z);

        FlowBatch {
            x,
            t,
            cond,
            u_target,
        }
    }
}
