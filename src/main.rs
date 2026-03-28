mod inspector;
mod layer;
mod loader;
mod loss;
mod optimizer;
mod sequential;
mod visualizer;

use ndarray::{Array2, Axis};
use ndarray_stats::QuantileExt;
use rand::seq::SliceRandom;
use std::fs;

use layer::Layer;
use layer::activation_layer::{Activation, ActivationLayer};
use layer::dense_layer::DenseLayer;
use layer::dropout_layer::DropoutLayer;
use loader::MnistDataset;
use loss::cross_entropy_with_logits;

use crate::optimizer::Adam;
use crate::sequential::Sequential;
use blas_src as _;

fn train() -> Result<(), Box<dyn std::error::Error>> {
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

    model.add(Layer::Dense(DenseLayer::new_random(784, 1024)));
    model.add(Layer::Activation(ActivationLayer::new(Activation::ReLU)));
    // model.add(Layer::Dropout(DropoutLayer::new(0.1)));

    model.add(Layer::Dense(DenseLayer::new_random(1024, 512)));
    model.add(Layer::Activation(ActivationLayer::new(Activation::ReLU)));
    model.add(Layer::Dropout(DropoutLayer::new(0.2)));

    model.add(Layer::Dense(DenseLayer::new_random(512, 10)));

    let epochs = 10;
    let batch_size = 64;
    let mut optimizer = Adam::new(0.001);
    let smoothing = 0.05;

    let n_samples = training_data.images.nrows();
    let mut indices: Vec<usize> = (0..n_samples).collect();

    for e in 0..epochs {
        indices.shuffle(&mut rand::rng());

        let mut total_train_loss = 0.0;
        let mut train_batch_count = 0;

        if (e + 1) % 15 == 0 {
            optimizer.learning_rate *= 0.5;
        }

        for batch_indices in indices.chunks(batch_size) {
            train_batch_count += 1;
            let batch_len = batch_indices.len();

            let batch_inputs = training_data.images.select(Axis(0), batch_indices);

            let mut targets = Array2::<f32>::from_elem((batch_len, 10), smoothing / 10.0);
            for (i, &idx) in batch_indices.iter().enumerate() {
                let label = training_data.labels[idx];
                targets[[i, label]] = 1.0 - smoothing + (smoothing / 10.0);
            }

            let (output, caches) = model.forward_training(&batch_inputs);
            let (loss_val, loss_grad) = cross_entropy_with_logits(&output, &targets);
            let l2_penalty = model.l2_loss();
            total_train_loss += loss_val + l2_penalty;

            let gradients = model.backward(&loss_grad, &caches);
            model.apply_gradients(&gradients, &mut optimizer);
        }

        let mut total_test_loss = 0.0;
        let mut misclassifications = 0;
        let n_test = test_data.images.nrows();
        let test_batch_size = 256;
        let test_chunks = (n_test as f32 / test_batch_size as f32).ceil() as usize;

        for start in (0..n_test).step_by(test_batch_size) {
            let end = (start + test_batch_size).min(n_test);
            let batch_inputs = test_data
                .images
                .slice(ndarray::s![start..end, ..])
                .to_owned();

            let current_batch_len = end - start;
            let mut test_targets = Array2::<f32>::zeros((current_batch_len, 10));
            for i in 0..current_batch_len {
                let label = test_data.labels[start + i];
                test_targets[[i, label]] = 1.0;
            }

            let predictions = model.forward(&batch_inputs);

            let (t_loss, _) = cross_entropy_with_logits(&predictions, &test_targets);
            total_test_loss += t_loss;

            for (i, row) in predictions.axis_iter(Axis(0)).enumerate() {
                if row.argmax().unwrap() != test_data.labels[start + i] {
                    misclassifications += 1;
                }
            }
        }

        let accuracy = (misclassifications as f32 / n_test as f32) * 100.0;

        println!(
            "Epoch {} - Train Loss: {:.4} | Test Loss: {:.4} | Test Error Rate: {:.2}%",
            e + 1,
            total_train_loss / train_batch_count as f32,
            total_test_loss / test_chunks as f32,
            accuracy
        );
    }

    let serialized_bytes: Vec<u8> = postcard::to_stdvec(&model)?;
    fs::write("model.postcard", serialized_bytes)?;

    Ok(())
}

fn analyze() -> Result<(), Box<dyn std::error::Error>> {
    let loaded_bytes = fs::read("model.postcard")?;
    let model: Sequential = postcard::from_bytes(&loaded_bytes)?;

    model.accept(&inspector::WeightDistributionInspector)?;
    model.accept(&inspector::HeatMapInspector)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // train()?;

    analyze()?;

    Ok(())
}
