#![allow(dead_code)]
use crate::layer::LayerGradients;
use ndarray::parallel::prelude::*;
use ndarray::{Array2, Zip}; // Import Rayon

#[derive(Clone, Debug)]
pub enum Activation {
    ReLU,
    LeakyReLU,
    Identity,
}

pub struct ActivationLayer {
    pub activation: Activation,
}

pub struct ActivationCache {
    input: Array2<f32>, // Pre-activation values
}

impl ActivationLayer {
    pub fn new(activation: Activation) -> Self {
        Self { activation }
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        match self.activation {
            Activation::ReLU => {
                let mut output = input.clone();
                output.par_mapv_inplace(|x| x.max(0.0));
                output
            }
            Activation::LeakyReLU => {
                let mut output = input.clone();
                output.par_mapv_inplace(|x| if x > 0.0 { x } else { 0.01 * x });
                output
            }
            Activation::Identity => input.clone(),
        }
    }

    pub fn forward_training(&self, input: &Array2<f32>) -> (Array2<f32>, ActivationCache) {
        let output = self.forward(input);
        (
            output,
            ActivationCache {
                input: input.clone(),
            },
        )
    }

    pub fn backward(
        &self,
        grad_output: &Array2<f32>,
        cache: &ActivationCache,
    ) -> (Array2<f32>, LayerGradients) {
        let grad_input = match self.activation {
            Activation::ReLU => {
                let mut g = grad_output.clone();
                Zip::from(&mut g)
                    .and(&cache.input)
                    .into_par_iter()
                    .for_each(|(go, &x)| {
                        if x <= 0.0 {
                            *go = 0.0;
                        }
                    });
                g
            }
            Activation::LeakyReLU => {
                let mut g = grad_output.clone();
                Zip::from(&mut g)
                    .and(&cache.input)
                    .into_par_iter()
                    .for_each(|(go, &x)| {
                        if x <= 0.0 {
                            *go *= 0.01;
                        }
                    });
                g
            }
            Activation::Identity => grad_output.clone(),
        };

        (grad_input, LayerGradients::None)
    }
}
