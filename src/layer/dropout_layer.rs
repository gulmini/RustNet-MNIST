use crate::layer::LayerGradients;
use ndarray::{Array2, Zip};
use ndarray_rand::{RandomExt, rand_distr};
use rand_distr::Bernoulli;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct DropoutLayer {
    drop_probability: f32,
}

pub struct DropoutCache {
    mask: Option<Array2<f32>>,
}

impl DropoutLayer {
    pub fn new(drop_probability: f32) -> Self {
        assert!(
            drop_probability >= 0.0 && drop_probability < 1.0,
            "Dropout probability must be in [0, 1)"
        );
        Self { drop_probability }
    }

    pub fn forward(&self, input: Array2<f32>) -> Array2<f32> {
        input
    }

    pub fn forward_training(&self, input: &Array2<f32>) -> (Array2<f32>, DropoutCache) {
        if self.drop_probability == 0.0 {
            return (input.clone(), DropoutCache { mask: None });
        }

        let keep_prob = 1.0 - self.drop_probability;
        let scale = 1.0 / keep_prob;
        let dist = Bernoulli::new(keep_prob.into()).unwrap();

        let mask = Array2::random(input.raw_dim(), dist).mapv(|b| if b { 1.0 } else { 0.0 });
        let mut output = input.clone();

        Zip::from(&mut output)
            .and(&mask)
            .for_each(|out, &m| *out *= m * scale);

        (output, DropoutCache { mask: Some(mask) })
    }

    pub fn backward(
        &self,
        grad_output: &Array2<f32>,
        cache: &DropoutCache,
    ) -> (Array2<f32>, LayerGradients) {
        let grad_input = match &cache.mask {
            Some(mask) => grad_output * mask * (1.0 / (1.0 - self.drop_probability)),
            None => grad_output.clone(),
        };
        (grad_input, LayerGradients::None)
    }
}
