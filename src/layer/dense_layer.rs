use crate::layer::LayerGradients;
use ndarray::{Array1, Array2, Axis, Zip};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct DenseLayer {
    input_size: usize,
    output_size: usize,
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    l2_lambda: Option<f32>,
}

pub struct DenseCache {
    input: Array2<f32>,
}

impl DenseLayer {
    pub fn new_random(input_size: usize, output_size: usize) -> Self {
        let limit = (6.0 / input_size as f32).sqrt();
        let weights = Array2::random(
            (input_size, output_size),
            Uniform::new(-limit, limit).unwrap(),
        );
        let biases = Array1::zeros(output_size);

        Self {
            input_size,
            output_size,
            weights,
            biases,
            l2_lambda: None,
        }
    }

    pub fn with_l2(mut self, lambda: f32) -> Self {
        self.l2_lambda = Some(lambda);
        self
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        input.dot(&self.weights) + &self.biases
    }

    pub fn forward_training(&self, input: &Array2<f32>) -> (Array2<f32>, DenseCache) {
        let output = self.forward(input);
        (
            output,
            DenseCache {
                input: input.clone(),
            },
        )
    }

    pub fn backward(
        &self,
        grad_output: &Array2<f32>,
        cache: &DenseCache,
    ) -> (Array2<f32>, LayerGradients) {
        let grad_input = grad_output.dot(&self.weights.t());
        let mut grad_weights = cache.input.t().dot(grad_output);
        let grad_biases = grad_output.sum_axis(Axis(0));

        if let Some(lambda) = self.l2_lambda {
            Zip::from(&mut grad_weights)
                .and(&self.weights)
                .for_each(|gw, &w| {
                    *gw += lambda * w;
                });
        }

        (
            grad_input,
            LayerGradients::Dense {
                weights: grad_weights,
                biases: grad_biases,
            },
        )
    }

    pub fn l2_loss(&self) -> f32 {
        if let Some(lambda) = self.l2_lambda {
            0.5 * lambda * self.weights.iter().map(|w| w * w).sum::<f32>()
        } else {
            0.0
        }
    }
}
