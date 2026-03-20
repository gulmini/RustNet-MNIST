#![allow(dead_code)]
pub mod dense_layer;
pub mod dropout_layer;

use std::any::Any;

use ndarray::{Array1, Array2};

pub trait Layer {
    fn shape(&self) -> (Option<usize>, Option<usize>);
    fn forward(&self, input: &Array2<f64>) -> Array2<f64>;
    fn forward_training(&self, input: &Array2<f64>) -> (Array2<f64>, Box<dyn Any>);
    fn backward(&self, grad_output: &Array2<f64>, cache: &dyn Any)
    -> (Array2<f64>, LayerGradients);
    fn apply_gradients(&mut self, gradients: &LayerGradients, learning_rate: f64);
}

pub struct LayerGradients {
    pub weights: Option<Array2<f64>>,
    pub biases: Option<Array1<f64>>,
}

impl LayerGradients {
    pub fn new(weights: Array2<f64>, biases: Array1<f64>) -> Self {
        Self {
            weights: Some(weights),
            biases: Some(biases),
        }
    }

    pub fn empty() -> Self {
        Self {
            weights: None,
            biases: None,
        }
    }
}

pub trait ActivationFn {
    fn f(x: f64) -> f64;
    fn df(x: f64) -> f64;
}

pub struct ReLU;
pub struct Sigmoid;
pub struct Tanh;
pub struct Identity;
pub struct LeakyReLU;

impl ActivationFn for ReLU {
    #[inline]
    fn f(x: f64) -> f64 {
        x.max(0.0)
    }
    fn df(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}

impl ActivationFn for Sigmoid {
    #[inline]
    fn f(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    fn df(x: f64) -> f64 {
        let sig = Self::f(x);
        sig * (1.0 - sig)
    }
}

impl ActivationFn for Tanh {
    #[inline]
    fn f(x: f64) -> f64 {
        x.tanh()
    }
    fn df(x: f64) -> f64 {
        1.0 - x.tanh().powi(2)
    }
}

impl ActivationFn for Identity {
    #[inline]
    fn f(x: f64) -> f64 {
        x
    }
    fn df(_x: f64) -> f64 {
        1.0
    }
}

impl ActivationFn for LeakyReLU {
    #[inline]
    fn f(x: f64) -> f64 {
        if x > 0.0 { x } else { 0.01 * x }
    }
    fn df(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.01 }
    }
}
