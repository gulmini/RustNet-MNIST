use crate::layer::LayerGradients;
use ndarray::Array2;

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
    input: Array2<f64>, // Pre-activation values
}

impl ActivationLayer {
    pub fn new(activation: Activation) -> Self {
        Self { activation }
    }

    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        match self.activation {
            Activation::ReLU => input.mapv(|x| x.max(0.0)),
            Activation::LeakyReLU => input.mapv(|x| if x > 0.0 { x } else { 0.01 * x }),
            Activation::Identity => input.clone(),
        }
    }

    pub fn forward_training(&self, input: &Array2<f64>) -> (Array2<f64>, ActivationCache) {
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
        grad_output: &Array2<f64>,
        cache: &ActivationCache,
    ) -> (Array2<f64>, LayerGradients) {
        let grad_input = match self.activation {
            Activation::ReLU => {
                let mut g = grad_output.clone();
                g.zip_mut_with(&cache.input, |go, &x| {
                    if x <= 0.0 {
                        *go = 0.0;
                    }
                });
                g
            }
            Activation::LeakyReLU => {
                let mut g = grad_output.clone();
                g.zip_mut_with(&cache.input, |go, &x| {
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
