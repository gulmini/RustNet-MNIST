pub mod activation_layer;
pub mod dense_layer;
pub mod dropout_layer;

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::inspector::LayerInspector;

#[derive(Serialize, Deserialize)]
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
            _ => None,
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

    pub fn get_params_mut(&mut self) -> Option<(&mut Array2<f32>, &mut Array1<f32>)> {
        match self {
            Layer::Dense(l) => Some((&mut l.weights, &mut l.biases)),
            _ => None,
        }
    }

    pub fn l2_loss(&self) -> f32 {
        match self {
            Layer::Dense(l) => l.l2_loss(),
            _ => 0.0,
        }
    }

    pub fn accept_inspection(
        &self,
        index: usize,
        inspector: &dyn LayerInspector,
    ) -> Result<(), Box<dyn std::error::Error>> {
        match self {
            Layer::Dense(l) => inspector.inspect_dense(index, l),
            Layer::Activation(l) => inspector.inspect_activation(index, l),
            Layer::Dropout(_) => Ok(()), // skip or add visit_dropout
        }
    }
}
