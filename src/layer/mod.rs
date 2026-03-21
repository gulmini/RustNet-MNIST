pub mod activation_layer;
pub mod dense_layer;
pub mod dropout_layer;

use crate::optimizer::Optimizer;
use ndarray::{Array1, Array2};

pub enum Layer {
    Dense(dense_layer::DenseLayer),
    Activation(activation_layer::ActivationLayer),
    Dropout(dropout_layer::DropoutLayer),
}

pub enum LayerCache {
    Dense(dense_layer::DenseCache),
    Activation(activation_layer::ActivationCache),
    Dropout(dropout_layer::DropoutCache),
}

pub enum LayerGradients {
    Dense {
        weights: Array2<f32>,
        biases: Array1<f32>,
    },
    None,
}

impl Layer {
    pub fn input_size(&self) -> Option<usize> {
        match self {
            Layer::Dense(l) => Some(l.input_size()),
            _ => None, // Activation and Dropout conform to their input shape
        }
    }

    pub fn output_size(&self) -> Option<usize> {
        match self {
            Layer::Dense(l) => Some(l.output_size()),
            _ => None,
        }
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            Layer::Dense(l) => l.forward(input),
            Layer::Activation(l) => l.forward(input),
            Layer::Dropout(l) => l.forward(input),
        }
    }

    pub fn forward_training(&self, input: &Array2<f32>) -> (Array2<f32>, LayerCache) {
        match self {
            Layer::Dense(l) => {
                let (out, cache) = l.forward_training(input);
                (out, LayerCache::Dense(cache))
            }
            Layer::Activation(l) => {
                let (out, cache) = l.forward_training(input);
                (out, LayerCache::Activation(cache))
            }
            Layer::Dropout(l) => {
                let (out, cache) = l.forward_training(input);
                (out, LayerCache::Dropout(cache))
            }
        }
    }

    pub fn backward(
        &self,
        grad_output: &Array2<f32>,
        cache: &LayerCache,
    ) -> (Array2<f32>, LayerGradients) {
        match (self, cache) {
            (Layer::Dense(l), LayerCache::Dense(c)) => l.backward(grad_output, c),
            (Layer::Activation(l), LayerCache::Activation(c)) => l.backward(grad_output, c),
            (Layer::Dropout(l), LayerCache::Dropout(c)) => l.backward(grad_output, c),
            _ => panic!("Layer cache type mismatch!"),
        }
    }

    pub fn apply_gradients(
        &mut self,
        layer_id: usize,
        gradients: &LayerGradients,
        optimizer: &mut dyn Optimizer,
    ) {
        if let (Layer::Dense(l), LayerGradients::Dense { weights, biases }) = (self, gradients) {
            l.apply_gradients(layer_id, weights, biases, optimizer);
        }
    }

    pub fn l2_loss(&self) -> f32 {
        match self {
            Layer::Dense(l) => l.l2_loss(),
            _ => 0.0, // Activations and Dropout don't have L2 loss
        }
    }
}
