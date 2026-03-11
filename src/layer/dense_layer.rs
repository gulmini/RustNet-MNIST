use crate::layer::LayerGradients;
use crate::optimizer::Optimizer;
use ndarray::{Array1, Array2, Axis};

#[derive(Debug)]
pub struct DenseLayer {
    input_size: usize,
    output_size: usize,
    weights: Array2<f64>,
    biases: Array1<f64>,
}

pub struct DenseCache {
    input: Array2<f64>,
}

impl DenseLayer {
    pub fn new_random(input_size: usize, output_size: usize) -> Self {
        let limit = (6.0 / input_size as f64).sqrt();
        let weights = Array2::from_shape_fn((input_size, output_size), |_| {
            (rand::random::<f64>() * 2.0 - 1.0) * limit
        });
        let biases = Array1::zeros(output_size);

        Self {
            input_size,
            output_size,
            weights,
            biases,
        }
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        input.dot(&self.weights) + &self.biases
    }

    pub fn forward_training(&self, input: &Array2<f64>) -> (Array2<f64>, DenseCache) {
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
        grad_output: &Array2<f64>,
        cache: &DenseCache,
    ) -> (Array2<f64>, LayerGradients) {
        let grad_input = grad_output.dot(&self.weights.t());
        let grad_weights = cache.input.t().dot(grad_output);
        let grad_biases = grad_output.sum_axis(Axis(0));

        (
            grad_input,
            LayerGradients::Dense {
                weights: grad_weights,
                biases: grad_biases,
            },
        )
    }

    pub fn apply_gradients(
        &mut self,
        w_grad: &Array2<f64>,
        b_grad: &Array1<f64>,
        optimizer: &mut dyn Optimizer,
    ) {
        optimizer.update_weights(&mut self.weights, w_grad);
        optimizer.update_biases(&mut self.biases, b_grad);
    }
}
