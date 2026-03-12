use std::any::Any;

use crate::layer::{Layer, LayerGradients};
use ndarray::Array2;

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

#[derive(Debug)]
pub struct SequentialTrace {
    pub layer_caches: Vec<Box<dyn Any>>,
}

impl Sequential {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) {
        // Skip layers with no size
        let last_known_output = self.layers.iter().rev().find_map(|l| l.shape().1);

        if let (Some(out_size), Some(in_size)) = (last_known_output, layer.shape().0) {
            assert_eq!(
                out_size, in_size,
                "layer size mismatch: previous output size {}, new layer input size {}",
                out_size, in_size
            );
        }

        self.layers.push(layer);
    }

    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        self.layers
            .iter()
            .fold(input.to_owned(), |acc, layer| layer.forward(&acc))
    }

    // Forward pass with caching of pre-activation values
    pub fn forward_training(&self, input: &Array2<f64>) -> (Array2<f64>, SequentialTrace) {
        let mut current_input = input.clone();
        let mut trace = SequentialTrace {
            layer_caches: Vec::new(),
        };

        for layer in &self.layers {
            let (output, cache) = layer.forward_training(&current_input);
            current_input = output;
            trace.layer_caches.push(cache);
        }

        (current_input, trace)
    }

    pub fn backward(
        &self,
        grad_output: &Array2<f64>,
        trace: &SequentialTrace,
    ) -> Vec<LayerGradients> {
        let mut current_grad = grad_output.clone();
        let mut all_grads = Vec::new();

        for (layer, cache) in self.layers.iter().zip(&trace.layer_caches).rev() {
            let (prev_grad, layer_grads) = layer.backward(&current_grad, cache.as_ref());
            current_grad = prev_grad;
            all_grads.push(layer_grads);
        }

        all_grads.reverse();
        all_grads
    }

    pub fn apply_gradients(&mut self, gradients: Vec<LayerGradients>, learning_rate: f64) {
        for (layer, grad) in self.layers.iter_mut().zip(gradients.iter()) {
            layer.apply_gradients(grad, learning_rate);
        }
    }
}
