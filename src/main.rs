mod layer;
mod loader;
mod loss;
mod optimizer;
mod sequential;
mod visualizer;

use ndarray::{Array2, Axis};
use ndarray_stats::QuantileExt;
use rand::seq::SliceRandom;

use layer::Layer;
use layer::activation_layer::{Activation, ActivationLayer};
use layer::dense_layer::DenseLayer;
use layer::dropout_layer::DropoutLayer;
use loader::MnistDataset;
use loss::cross_entropy_with_logits;
use optimizer::SGD;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let training_data = MnistDataset::load(
        "dataset/train-images-idx3-ubyte",
        "dataset/train-labels-idx1-ubyte",
    )?;

    let test_data = MnistDataset::load(
        "dataset/t10k-images-idx3-ubyte",
        "dataset/t10k-labels-idx1-ubyte",
    )?;

    println!("Training data loaded");

    let mut model = sequential::Sequential::new();

    // Layer 1
    model.add(Layer::Dense(DenseLayer::new_random(784, 512)));
    model.add(Layer::Activation(ActivationLayer::new(
        Activation::LeakyReLU,
    )));
    model.add(Layer::Dropout(DropoutLayer::new(0.2)));

    // Layer 2
    model.add(Layer::Dense(DenseLayer::new_random(512, 256)));
    model.add(Layer::Activation(ActivationLayer::new(
        Activation::LeakyReLU,
    )));
    model.add(Layer::Dropout(DropoutLayer::new(0.2)));

    // Layer 3 (Output logits, Softmax handled by loss function)
    model.add(Layer::Dense(DenseLayer::new_random(256, 10)));

    let epochs = 20;
    let batch_size = 128;
    let mut optimizer = SGD::new(0.1);

    let n_samples = training_data.images.nrows();
    let mut indices: Vec<usize> = (0..n_samples).collect();

    for e in 0..epochs {
        indices.shuffle(&mut rand::rng());

        for batch_indices in indices.chunks(batch_size) {
            let batch_len = batch_indices.len();

            let mut batch_inputs = Array2::<f64>::zeros((batch_len, 784));
            let mut targets = Array2::<f64>::zeros((batch_len, 10));

            for (i, &idx) in batch_indices.iter().enumerate() {
                batch_inputs
                    .row_mut(i)
                    .assign(&training_data.images.row(idx));
                let label = training_data.labels[idx];
                targets[[i, label]] = 1.0;
            }

            let (output, caches) = model.forward_training(&batch_inputs);

            // Clean Loss Abstraction
            let (_loss, loss_grad) = cross_entropy_with_logits(&output, &targets);

            let gradients = model.backward(&loss_grad, &caches);
            model.apply_gradients(&gradients, &mut optimizer);
        }

        optimizer.learning_rate *= 0.99;
        println!(
            "epoch {} complete. learning_rate {:.5}",
            e + 1,
            optimizer.learning_rate
        );
    }

    let mut misclassifications: usize = 0;
    let test_batch_size = 256;
    let n_test = test_data.images.nrows();

    for start in (0..n_test).step_by(test_batch_size) {
        let end = (start + test_batch_size).min(n_test);
        let batch_inputs = test_data
            .images
            .slice(ndarray::s![start..end, ..])
            .to_owned();

        let predictions = model.forward(&batch_inputs);

        for (i, row) in predictions.axis_iter(Axis(0)).enumerate() {
            let prediction = row.argmax().unwrap();
            if prediction != test_data.labels[start + i] {
                misclassifications += 1;
            }
        }
    }

    println!(
        "Error Rate: {:.2}%",
        (misclassifications as f64 / n_test as f64) * 100.0
    );

    Ok(())
}
