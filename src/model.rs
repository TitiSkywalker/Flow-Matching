use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
        interpolate::{Interpolate2d, Interpolate2dConfig, InterpolateMode},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    prelude::*,
};

pub fn time_embedding<B: Backend>(
    ts: &Tensor<B, 1>,
    output_dim: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    assert!(
        output_dim % 2 == 0,
        "output_dim must be even for this implementation"
    );

    let batch_size = ts.shape().dims[0];
    let half = output_dim / 2;

    // Match PyTorch: i range length = output_dim
    let i: Tensor<B, 1> = Tensor::arange(0..output_dim as i64, device).float();

    // log_term uses output_dim as in your PyTorch code
    let log_term = -10000f32.ln() / output_dim as f32;

    // div_term length = output_dim
    let div_term: Tensor<B, 1> = (i * log_term).exp();

    // angle shape: (batch, output_dim)
    let angle: Tensor<B, 2> = ts.clone().unsqueeze_dim(1) * div_term.unsqueeze_dim(0);

    // reshape to (batch, half, 2) so columns are pairs (0,1), (2,3), ...
    let angle_pairs: Tensor<B, 3> = angle.reshape(Shape::new([batch_size, half, 2]));

    // sin on first of each pair, cos on second of each pair
    let sin_part: Tensor<B, 2> = angle_pairs
        .clone()
        .select(2, Tensor::<B, 1, Int>::from_ints([0], device))
        .squeeze_dim(2)
        .sin();
    let cos_part: Tensor<B, 2> = angle_pairs
        .select(2, Tensor::<B, 1, Int>::from_ints([1], device))
        .squeeze_dim(2)
        .cos();

    // stack them into (batch, half, 2)
    let s3: Tensor<B, 3> = sin_part.unsqueeze_dim(2);
    let c3: Tensor<B, 3> = cos_part.unsqueeze_dim(2);
    let sc: Tensor<B, 3> = Tensor::cat([s3, c3].to_vec(), 2);

    // back to (batch, output_dim)
    sc.reshape(Shape::new([batch_size, output_dim]))
}

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    norm2: BatchNorm<B>,
    relu: Relu,
    // embedding MLPs
    time_lin1: Linear<B>,
    time_lin2: Linear<B>,
    cond_lin1: Linear<B>,
    cond_lin2: Linear<B>,
}

impl<B: Backend> ConvBlock<B> {
    /// init helper
    pub fn init(
        in_channels: usize,
        out_channels: usize,
        time_embed_dim: usize,
        cond_embed_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            conv1: Conv2dConfig::new([in_channels, out_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            conv2: Conv2dConfig::new([out_channels, out_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            norm2: BatchNormConfig::new(out_channels).init(device),
            relu: Relu::new(),
            time_lin1: LinearConfig::new(time_embed_dim, in_channels).init(device),
            time_lin2: LinearConfig::new(in_channels, in_channels).init(device),
            cond_lin1: LinearConfig::new(cond_embed_dim, in_channels).init(device),
            cond_lin2: LinearConfig::new(in_channels, in_channels).init(device),
        }
    }

    /// forward
    /// - x: [B, C, H, W]
    /// - time_emb: [B, time_embed_dim]
    /// - cond_emb: [B, cond_embed_dim]
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
        time_emb: Tensor<B, 2>,
        cond_emb: Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        // project embeddings -> [B, C]
        let t = self.time_lin1.forward(time_emb);
        let t = self.relu.forward(t);
        let t = self.time_lin2.forward(t);
        let c = self.cond_lin1.forward(cond_emb);
        let c = self.relu.forward(c);
        let c = self.cond_lin2.forward(c);

        // reshape to [B, C, 1, 1] and broadcast add
        let batch = x.dims()[0];
        let in_channels = x.dims()[1];
        let t_reshaped = t.reshape([batch, in_channels, 1, 1]);
        let c_reshaped = c.reshape([batch, in_channels, 1, 1]);

        let x = x + t_reshaped + c_reshaped;

        let x = self.conv1.forward(x);
        let x = self.relu.forward(x);
        let x = self.conv2.forward(x);
        let x = self.norm2.forward(x);
        self.relu.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    down1: ConvBlock<B>,
    down2: ConvBlock<B>,
    bottom: ConvBlock<B>,
    up2: ConvBlock<B>,
    up1: ConvBlock<B>,
    out_conv: Conv2d<B>,
    pool: MaxPool2d,
    upsample: Interpolate2d,
}

impl<B: Backend> UNet<B> {
    pub fn init(
        in_channels: usize,
        time_embed_dim: usize,
        cond_embed_dim: usize,
        device: &B::Device,
    ) -> Self {
        let down1 = ConvBlock::init(in_channels, 64, time_embed_dim, cond_embed_dim, device);
        let down2 = ConvBlock::init(64, 128, time_embed_dim, cond_embed_dim, device);
        let bottom = ConvBlock::init(128, 256, time_embed_dim, cond_embed_dim, device);
        let up2 = ConvBlock::init(128 + 256, 128, time_embed_dim, cond_embed_dim, device);
        let up1 = ConvBlock::init(128 + 64, 64, time_embed_dim, cond_embed_dim, device);
        let out_conv = Conv2dConfig::new([64, in_channels], [1, 1]).init(device);
        let pool = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        // Bilinear interpolation
        let upsample = Interpolate2dConfig::new()
            .with_mode(InterpolateMode::Nearest)
            .with_scale_factor(Some([2.0, 2.0]))
            .init();

        Self {
            down1,
            down2,
            bottom,
            up2,
            up1,
            out_conv,
            pool,
            upsample,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 4>,        // [B, C, H, W]
        time_emb: Tensor<B, 2>, // [B, time_embed_dim]
        cond_emb: Tensor<B, 2>, // [B, cond_embed_dim]
    ) -> Tensor<B, 4> {
        let x1 = self.down1.forward(x, time_emb.clone(), cond_emb.clone());
        let x = self.pool.forward(x1.clone());

        let x2 = self.down2.forward(x, time_emb.clone(), cond_emb.clone());
        let x = self.pool.forward(x2.clone());

        let x = self.bottom.forward(x, time_emb.clone(), cond_emb.clone());

        // Upsample + skip connections
        let x = self.upsample.forward(x);
        let x = Tensor::cat([x, x2].to_vec(), 1);
        let x = self.up2.forward(x, time_emb.clone(), cond_emb.clone());

        let x = self.upsample.forward(x);
        let x = Tensor::cat([x, x1].to_vec(), 1);
        let x = self.up1.forward(x, time_emb.clone(), cond_emb.clone());

        self.out_conv.forward(x)
    }
}
