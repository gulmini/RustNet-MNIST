use crate::layer::{Layer, LayerCache, LayerGradients};
use crate::optimizer::Optimizer;
use ndarray::Array2;

pub struct Sequential {
    layers: Vec<Layer>,
}

impl Sequential {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add(&mut self, layer: Layer) {
        // If the new layer has a specific input size requirement...
        if let Some(new_input_size) = layer.input_size() {
            // Find the output size of the most recently added layer that has a fixed output size
            let last_output_size = self.layers.iter().rev().find_map(|l| l.output_size());

            if let Some(expected_size) = last_output_size {
                assert_eq!(
                    expected_size, new_input_size,
                    "Architecture Mismatch: Cannot add layer with input size {} after a layer with output size {}",
                    new_input_size, expected_size
                );
            }
        }

        self.layers.push(layer);
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        self.layers
            .iter()
            .fold(input.to_owned(), |acc, layer| layer.forward(&acc))
    }

    pub fn forward_training(&self, input: &Array2<f32>) -> (Array2<f32>, Vec<LayerCache>) {
        let mut current_input = input.clone();
        let mut caches = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let (output, cache) = layer.forward_training(&current_input);
            current_input = output;
            caches.push(cache);
        }

        (current_input, caches)
    }

    pub fn backward(
        &self,
        grad_output: &Array2<f32>,
        caches: &[LayerCache],
    ) -> Vec<LayerGradients> {
        let mut current_grad = grad_output.clone();
        let mut all_grads = Vec::with_capacity(self.layers.len());

        for (layer, cache) in self.layers.iter().zip(caches).rev() {
            let (prev_grad, layer_grads) = layer.backward(&current_grad, cache);
            current_grad = prev_grad;
            all_grads.push(layer_grads);
        }

        all_grads.reverse();
        all_grads
    }

    pub fn apply_gradients(&mut self, gradients: &[LayerGradients], optimizer: &mut dyn Optimizer) {
        optimizer.step();

        // Enumerate provides stable `layer_id`
        for (layer_id, (layer, grad)) in self.layers.iter_mut().zip(gradients.iter()).enumerate() {
            layer.apply_gradients(layer_id, grad, optimizer);
        }
    }

    pub fn l2_loss(&self) -> f32 {
        self.layers.iter().map(|layer| layer.l2_loss()).sum()
    }
}
