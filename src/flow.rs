use burn::tensor::{Distribution, Tensor, backend::Backend};

pub fn alpha<B: Backend>(t: &Tensor<B, 1>) -> Tensor<B, 1> {
    t.clone()
}

pub fn alpha_prime<B: Backend>(t: &Tensor<B, 1>) -> Tensor<B, 1> {
    Tensor::ones_like(t)
}

pub fn beta<B: Backend>(t: &Tensor<B, 1>) -> Tensor<B, 1> {
    Tensor::ones_like(t) - t.clone()
}

pub fn beta_prime<B: Backend>(t: &Tensor<B, 1>) -> Tensor<B, 1> {
    -Tensor::ones_like(t)
}

pub fn gaussian_vector_field<B: Backend>(
    t: &Tensor<B, 1>, // (B,)
    z: &Tensor<B, 4>, // (B, C, W, H)
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    // Random noise same shape as z
    let epsilon = Tensor::random_like(z, Distribution::Normal(0.0, 1.0));

    // Compute alpha, beta, etc.
    let a = alpha(t);
    let b = beta(t);
    let a_p = alpha_prime(t);
    let b_p = beta_prime(t);

    // Expand to match (B, W, H)
    // => unsqueeze and broadcast
    let a = a.unsqueeze_dims(&[1, 1, 1]);
    let b = b.unsqueeze_dims(&[1, 1, 1]);
    let a_p = a_p.unsqueeze_dims(&[1, 1, 1]);
    let b_p = b_p.unsqueeze_dims(&[1, 1, 1]);

    // Compute x, u
    let x = a.clone() * z.clone() + b.clone() * epsilon.clone();
    let u = a_p * z.clone() + b_p * epsilon;

    (x, u)
}
