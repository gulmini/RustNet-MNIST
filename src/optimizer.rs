use ndarray::{Array1, Array2};

pub trait Optimizer {
    fn update_weights(&mut self, weights: &mut Array2<f64>, grad: &Array2<f64>);
    fn update_biases(&mut self, biases: &mut Array1<f64>, grad: &Array1<f64>);
}

pub struct SGD {
    pub learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn update_weights(&mut self, weights: &mut Array2<f64>, grad: &Array2<f64>) {
        *weights = &*weights - &(grad * self.learning_rate);
    }

    fn update_biases(&mut self, biases: &mut Array1<f64>, grad: &Array1<f64>) {
        *biases = &*biases - &(grad * self.learning_rate);
    }
}
