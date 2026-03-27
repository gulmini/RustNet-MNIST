#![allow(dead_code)]
use ndarray::{Array1, Array2, Zip};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub trait Optimizer {
    fn step(&mut self) {}

    fn apply(
        &mut self,
        weights: Vec<&mut Array2<f32>>,
        biases: Vec<&mut Array1<f32>>,
        weight_grads: Vec<&Array2<f32>>,
        bias_grads: Vec<&Array1<f32>>,
    );
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
    fn apply(
        &mut self,
        mut weights: Vec<&mut Array2<f32>>,
        mut biases: Vec<&mut Array1<f32>>,
        weight_grads: Vec<&Array2<f32>>,
        bias_grads: Vec<&Array1<f32>>,
    ) {
        let lr = self.learning_rate;

        for (w, g) in weights.iter_mut().zip(weight_grads.iter()) {
            Zip::from(&mut **w)
                .and(*g)
                .into_par_iter()
                .for_each(|(wi, &gi)| *wi -= lr * gi);
        }

        for (b, g) in biases.iter_mut().zip(bias_grads.iter()) {
            Zip::from(&mut **b)
                .and(*g)
                .into_par_iter()
                .for_each(|(bi, &gi)| *bi -= lr * gi);
        }
    }
}

pub struct Momentum {
    pub learning_rate: f32,
    pub momentum: f32,
    v_weights: Vec<Array2<f32>>,
    v_biases: Vec<Array1<f32>>,
}

impl Momentum {
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            v_weights: Vec::new(),
            v_biases: Vec::new(),
        }
    }
}

impl Optimizer for Momentum {
    fn apply(
        &mut self,
        mut weights: Vec<&mut Array2<f32>>,
        mut biases: Vec<&mut Array1<f32>>,
        weight_grads: Vec<&Array2<f32>>,
        bias_grads: Vec<&Array1<f32>>,
    ) {
        // Lazy initialization on the first pass
        if self.v_weights.is_empty() {
            for w in &weights {
                self.v_weights.push(Array2::zeros(w.raw_dim()));
            }
            for b in &biases {
                self.v_biases.push(Array1::zeros(b.raw_dim()));
            }
        }

        let momentum = self.momentum;
        let lr = self.learning_rate;

        for ((w, g), v) in weights
            .iter_mut()
            .zip(weight_grads.iter())
            .zip(self.v_weights.iter_mut())
        {
            Zip::from(&mut **w)
                .and(v)
                .and(*g)
                .into_par_iter()
                .for_each(|(wi, vi, &gi)| {
                    *vi = momentum * (*vi) + lr * gi;
                    *wi -= *vi;
                });
        }

        for ((b, g), v) in biases
            .iter_mut()
            .zip(bias_grads.iter())
            .zip(self.v_biases.iter_mut())
        {
            Zip::from(&mut **b)
                .and(v)
                .and(*g)
                .into_par_iter()
                .for_each(|(bi, vi, &gi)| {
                    *vi = momentum * (*vi) + lr * gi;
                    *bi -= *vi;
                });
        }
    }
}

pub struct Adam {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub t: i32,
    m_weights: Vec<Array2<f32>>,
    v_weights: Vec<Array2<f32>>,
    m_biases: Vec<Array1<f32>>,
    v_biases: Vec<Array1<f32>>,
}

impl Adam {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            m_weights: Vec::new(),
            v_weights: Vec::new(),
            m_biases: Vec::new(),
            v_biases: Vec::new(),
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.t += 1;
    }

    fn apply(
        &mut self,
        mut weights: Vec<&mut Array2<f32>>,
        mut biases: Vec<&mut Array1<f32>>,
        weight_grads: Vec<&Array2<f32>>,
        bias_grads: Vec<&Array1<f32>>,
    ) {
        // Lazy initialization
        if self.m_weights.is_empty() {
            for w in &weights {
                self.m_weights.push(Array2::zeros(w.raw_dim()));
                self.v_weights.push(Array2::zeros(w.raw_dim()));
            }
            for b in &biases {
                self.m_biases.push(Array1::zeros(b.raw_dim()));
                self.v_biases.push(Array1::zeros(b.raw_dim()));
            }
        }

        let b1 = self.beta1;
        let b2 = self.beta2;
        let eps = self.epsilon;
        let lr = self.learning_rate;
        let c1 = 1.0 - b1.powi(self.t);
        let c2 = 1.0 - b2.powi(self.t);

        for (((w, g), m), v) in weights
            .iter_mut()
            .zip(weight_grads.iter())
            .zip(self.m_weights.iter_mut())
            .zip(self.v_weights.iter_mut())
        {
            Zip::from(&mut **w)
                .and(m)
                .and(v)
                .and(*g)
                .into_par_iter()
                .for_each(|(wi, mi, vi, &gi)| {
                    *mi = b1 * (*mi) + (1.0 - b1) * gi;
                    *vi = b2 * (*vi) + (1.0 - b2) * gi * gi;
                    let m_hat = *mi / c1;
                    let v_hat = *vi / c2;
                    *wi -= lr * m_hat / (v_hat.sqrt() + eps);
                });
        }

        for (((b, g), m), v) in biases
            .iter_mut()
            .zip(bias_grads.iter())
            .zip(self.m_biases.iter_mut())
            .zip(self.v_biases.iter_mut())
        {
            Zip::from(&mut **b)
                .and(m)
                .and(v)
                .and(*g)
                .into_par_iter()
                .for_each(|(bi, mi, vi, &gi)| {
                    *mi = b1 * (*mi) + (1.0 - b1) * gi;
                    *vi = b2 * (*vi) + (1.0 - b2) * gi * gi;
                    let m_hat = *mi / c1;
                    let v_hat = *vi / c2;
                    *bi -= lr * m_hat / (v_hat.sqrt() + eps);
                });
        }
    }
}
