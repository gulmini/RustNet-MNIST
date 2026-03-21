#![allow(dead_code)]
use std::collections::HashMap;

use ndarray::{Array1, Array2, Zip};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub trait Optimizer {
    fn step(&mut self) {}
    fn update_weights(&mut self, layer_id: usize, weights: &mut Array2<f32>, grad: &Array2<f32>);
    fn update_biases(&mut self, layer_id: usize, biases: &mut Array1<f32>, grad: &Array1<f32>);
}

pub struct SGD {
    pub learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn update_weights(&mut self, _layer_id: usize, weights: &mut Array2<f32>, grad: &Array2<f32>) {
        let lr = self.learning_rate;
        Zip::from(weights).and(grad).for_each(|w, &g| *w -= lr * g);
    }

    fn update_biases(&mut self, _layer_id: usize, biases: &mut Array1<f32>, grad: &Array1<f32>) {
        let lr = self.learning_rate;
        Zip::from(biases).and(grad).for_each(|b, &g| *b -= lr * g);
    }
}

pub struct Momentum {
    pub learning_rate: f32,
    pub momentum: f32,
    velocity_w: HashMap<usize, Array2<f32>>,
    velocity_b: HashMap<usize, Array1<f32>>,
}

impl Momentum {
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity_w: HashMap::new(),
            velocity_b: HashMap::new(),
        }
    }
}

impl Optimizer for Momentum {
    fn update_weights(&mut self, layer_id: usize, weights: &mut Array2<f32>, grad: &Array2<f32>) {
        let v = self
            .velocity_w
            .entry(layer_id)
            .or_insert_with(|| Array2::zeros(weights.raw_dim()));
        let momentum = self.momentum;
        let lr = self.learning_rate;

        Zip::from(weights).and(v).and(grad).for_each(|w, vi, &g| {
            *vi = momentum * (*vi) + lr * g;
            *w -= *vi;
        });
    }

    fn update_biases(&mut self, layer_id: usize, biases: &mut Array1<f32>, grad: &Array1<f32>) {
        let v = self
            .velocity_b
            .entry(layer_id)
            .or_insert_with(|| Array1::zeros(biases.raw_dim()));
        let momentum = self.momentum;
        let lr = self.learning_rate;

        Zip::from(biases).and(v).and(grad).for_each(|b, vi, &g| {
            *vi = momentum * (*vi) + lr * g;
            *b -= *vi;
        });
    }
}

pub struct Adam {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub t: i32,
    m_w: HashMap<usize, Array2<f32>>,
    v_w: HashMap<usize, Array2<f32>>,
    m_b: HashMap<usize, Array1<f32>>,
    v_b: HashMap<usize, Array1<f32>>,
}

impl Adam {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            m_w: HashMap::new(),
            v_w: HashMap::new(),
            m_b: HashMap::new(),
            v_b: HashMap::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.t += 1;
    }

    fn update_weights(&mut self, layer_id: usize, weights: &mut Array2<f32>, grad: &Array2<f32>) {
        let m = self
            .m_w
            .entry(layer_id)
            .or_insert_with(|| Array2::zeros(weights.raw_dim()));
        let v = self
            .v_w
            .entry(layer_id)
            .or_insert_with(|| Array2::zeros(weights.raw_dim()));

        let b1 = self.beta1;
        let b2 = self.beta2;
        let eps = self.epsilon;
        let lr = self.learning_rate;

        let c1 = 1.0 - b1.powi(self.t);
        let c2 = 1.0 - b2.powi(self.t);

        Zip::from(weights)
            .and(m)
            .and(v)
            .and(grad)
            .into_par_iter()
            .for_each(|(w, mi, vi, &g)| {
                *mi = b1 * (*mi) + (1.0 - b1) * g;
                *vi = b2 * (*vi) + (1.0 - b2) * g * g;

                let m_hat = *mi / c1;
                let v_hat = *vi / c2;

                *w -= lr * m_hat / (v_hat.sqrt() + eps);
            });
    }

    fn update_biases(&mut self, layer_id: usize, biases: &mut Array1<f32>, grad: &Array1<f32>) {
        let m = self
            .m_b
            .entry(layer_id)
            .or_insert_with(|| Array1::zeros(biases.raw_dim()));
        let v = self
            .v_b
            .entry(layer_id)
            .or_insert_with(|| Array1::zeros(biases.raw_dim()));

        let b1 = self.beta1;
        let b2 = self.beta2;
        let eps = self.epsilon;
        let lr = self.learning_rate;
        let c1 = 1.0 - b1.powi(self.t);
        let c2 = 1.0 - b2.powi(self.t);

        Zip::from(biases)
            .and(m)
            .and(v)
            .and(grad)
            .into_par_iter()
            .for_each(|(b, mi, vi, &g)| {
                *mi = b1 * (*mi) + (1.0 - b1) * g;
                *vi = b2 * (*vi) + (1.0 - b2) * g * g;

                let m_hat = *mi / c1;
                let v_hat = *vi / c2;

                *b -= lr * m_hat / (v_hat.sqrt() + eps);
            });
    }
}
