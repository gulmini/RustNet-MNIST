use crate::layer::LayerGradients;
use ndarray::Array2;
use ndarray_rand::{RandomExt, rand_distr};
use rand_distr::Bernoulli;

#[derive(Debug)]
pub struct DropoutLayer {
    drop_probability: f64,
}

pub struct DropoutCache {
    mask: Option<Array2<f64>>,
}

impl DropoutLayer {
    pub fn new(drop_probability: f64) -> Self {
        assert!(
            drop_probability >= 0.0 && drop_probability < 1.0,
            "Dropout probability must be in [0, 1)"
        );
        Self { drop_probability }
    }

    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        input.clone()
    }

    pub fn forward_training(&self, input: &Array2<f64>) -> (Array2<f64>, DropoutCache) {
        if self.drop_probability == 0.0 {
            return (input.clone(), DropoutCache { mask: None });
        }

        let keep_prob = 1.0 - self.drop_probability;
        let scale = 1.0 / keep_prob;
        let dist = Bernoulli::new(keep_prob).unwrap();

        // Vectorized boolean mask generation directly translated to f64 mask map
        let mask = Array2::random(input.raw_dim(), dist).mapv(|b| if b { 1.0 } else { 0.0 });

        let output = input * &mask * scale;
        (output, DropoutCache { mask: Some(mask) })
    }

    pub fn backward(
        &self,
        grad_output: &Array2<f64>,
        cache: &DropoutCache,
    ) -> (Array2<f64>, LayerGradients) {
        let grad_input = match &cache.mask {
            Some(mask) => grad_output * mask * (1.0 / (1.0 - self.drop_probability)),
            None => grad_output.clone(),
        };
        (grad_input, LayerGradients::None)
    }
}
