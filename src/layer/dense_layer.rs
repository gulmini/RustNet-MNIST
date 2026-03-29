use crate::layer::LayerGradients;
use ndarray::{Array, Array1, Array2, Axis, Dimension, Zip};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum ParamState<D: Dimension> {
    Float(Array<f32, D>),
    Quantized { data: Array<f32, D>, bits: u8 },
}

impl<D: Dimension> ParamState<D> {
    pub fn get_effective(&self) -> Array<f32, D> {
        match self {
            ParamState::Float(w) => w.to_owned(),
            ParamState::Quantized { data, bits } => {
                let min = -0.1_f32;
                let max = 0.1_f32;
                let range = max - min;
                let levels = ((1_u32 << bits) - 1) as f32;

                let scale_in = levels / range;
                let scale_out = range / levels;

                let mut result = data.clone();

                result.par_mapv_inplace(|w| {
                    let clamped = w.clamp(min, max);
                    ((clamped - min) * scale_in).round() * scale_out + min
                });

                result
            }
        }
    }

    pub fn raw(&self) -> &Array<f32, D> {
        match self {
            ParamState::Float(w) => w,
            ParamState::Quantized { data, .. } => data,
        }
    }

    pub fn raw_mut(&mut self) -> &mut Array<f32, D> {
        match self {
            ParamState::Float(w) => w,
            ParamState::Quantized { data, .. } => data,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct DenseLayer {
    input_size: usize,
    output_size: usize,
    pub weights: ParamState<ndarray::Ix2>,
    pub biases: ParamState<ndarray::Ix1>,
    l2_lambda: Option<f32>,
}

pub struct DenseCache {
    input: Array2<f32>,
    q_weights: Array2<f32>,
}

impl DenseLayer {
    pub fn new_random(input_size: usize, output_size: usize) -> Self {
        let limit = (6.0 / input_size as f32).sqrt();
        let weights_array = Array2::random(
            (input_size, output_size),
            Uniform::new(-limit, limit).unwrap(),
        );
        let biases = Array1::zeros(output_size);

        Self {
            input_size,
            output_size,
            weights: ParamState::Float(weights_array),
            biases: ParamState::Float(biases),
            l2_lambda: None,
        }
    }

    pub fn with_l2(mut self, lambda: f32) -> Self {
        self.l2_lambda = Some(lambda);
        self
    }

    pub fn with_quantization(mut self, bits: u8) -> Self {
        if let ParamState::Float(w) = self.weights {
            self.weights = ParamState::Quantized { data: w, bits };
        }
        if let ParamState::Float(b) = self.biases {
            self.biases = ParamState::Quantized { data: b, bits };
        }
        self
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    pub fn get_effective_weights(&self) -> Array2<f32> {
        self.weights.get_effective()
    }

    pub fn get_effective_biases(&self) -> Array1<f32> {
        self.biases.get_effective()
    }

    pub fn forward(&self, input: Array2<f32>) -> Array2<f32> {
        let w = self.get_effective_weights();
        let b = self.get_effective_biases();
        input.dot(&w) + &b
    }

    pub fn forward_training(&self, input: &Array2<f32>) -> (Array2<f32>, DenseCache) {
        let w = self.get_effective_weights();
        let b = self.get_effective_biases();
        let output = input.dot(&w) + &b;
        (
            output,
            DenseCache {
                input: input.clone(),
                q_weights: w,
            },
        )
    }

    pub fn backward(
        &self,
        grad_output: &Array2<f32>,
        cache: &DenseCache,
    ) -> (Array2<f32>, LayerGradients) {
        let grad_input = grad_output.dot(&cache.q_weights.t());
        let mut grad_weights = cache.input.t().dot(grad_output);
        let grad_biases = grad_output.sum_axis(Axis(0));

        if let Some(lambda) = self.l2_lambda {
            let raw_w = self.weights.raw();
            Zip::from(&mut grad_weights).and(raw_w).for_each(|gw, &w| {
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
            0.5 * lambda * self.weights.raw().iter().map(|w| w * w).sum::<f32>()
        } else {
            0.0
        }
    }
}
