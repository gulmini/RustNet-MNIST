use crate::layer::activation_layer::ActivationLayer;
use crate::layer::dense_layer::DenseLayer;

pub trait LayerInspector {
    fn inspect_dense(&mut self, index: usize, layer: &DenseLayer) {}
    fn inspect_activation(&mut self, index: usize, layer: &ActivationLayer) {}
}

pub struct WeightInspector;

impl LayerInspector for WeightInspector {
    fn inspect_dense(&mut self, _index: usize, layer: &DenseLayer) {
        let print_stats = |name: &str, mut data: Vec<f32>| {
            data.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = data.len() as f32;
            let mean = data.iter().sum::<f32>() / n;
            let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
            let std_dev = variance.sqrt();
            let q1 = data[(n * 0.25) as usize];
            let q2 = data[(n * 0.5) as usize];
            let q3 = data[(n * 0.75) as usize];
            println!(
                "{}: mean={}, q1={}, median={}, q3={}, std_dev={}",
                name, mean, q1, q2, q3, std_dev
            );
        };

        print_stats("Weights", layer.weights.iter().copied().collect());
        print_stats("Bias", layer.biases.iter().copied().collect());
    }
}
