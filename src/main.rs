mod layer;
mod loader;
mod sequential;
mod visualizer;
use ndarray::{Array1, Array2, Axis, stack};
use ndarray_stats::QuantileExt;
use rand;

use loader::MnistDataset;

use rand::seq::SliceRandom;
// use visualizer::visualize_image;

use layer::dense_layer::DenseLayer;
use layer::dropout_layer::DropoutLayer;
use layer::{Identity, LeakyReLU};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let training_images = MnistDataset::load(
        "dataset/train-images-idx3-ubyte",
        "dataset/train-labels-idx1-ubyte",
    )?
    .images;

    let test_images = MnistDataset::load(
        "dataset/t10k-images-idx3-ubyte",
        "dataset/t10k-labels-idx1-ubyte",
    )?
    .images;

    let mut model = sequential::Sequential::new();
    let layer1 = DenseLayer::<LeakyReLU>::new_random(784, 512);
    let dropout1 = DropoutLayer::new(0.2);
    let layer2 = DenseLayer::<LeakyReLU>::new_random(512, 256);
    let dropout2 = DropoutLayer::new(0.2);
    let layer3 = DenseLayer::<Identity>::new_random(256, 10);

    model.add_layer(Box::new(layer1));
    model.add_layer(Box::new(dropout1));
    model.add_layer(Box::new(layer2));
    model.add_layer(Box::new(dropout2));
    model.add_layer(Box::new(layer3));

    let mut training_images = training_images
        .iter()
        .map(|img| {
            (
                Array1::from_vec(img.data.iter().map(|&b| b as f64 / 255.0).collect()),
                img.label,
            )
        })
        .collect::<Vec<(Array1<f64>, u8)>>();

    let test_images = test_images
        .iter()
        .map(|img| {
            (
                Array1::from_vec(img.data.iter().map(|&b| b as f64 / 255.0).collect()),
                img.label,
            )
        })
        .collect::<Vec<(Array1<f64>, u8)>>();

    // Naive SGD with mini-batching, trained over many epochs with reshuffling
    let epochs = 30;
    let batch_size = 64;
    let mut learning_rate = 0.1;

    for e in 0..epochs {
        for batch in training_images.chunks(batch_size) {
            let batch_len = batch.len() as f64;

            let batch_inputs = stack(
                Axis(0),
                &batch
                    .iter()
                    .map(|(data, _)| data.view())
                    .collect::<Vec<_>>(),
            )?;

            let targets = Array2::from_shape_fn((batch.len(), 10), |(i, j)| {
                if batch[i].1 as usize == j { 1.0 } else { 0.0 }
            });

            let (output, trace) = model.forward_training(&batch_inputs);

            let max_per_row = output.map_axis(Axis(1), |row| {
                row.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            });

            // Probabilities expressed as Softmax of output
            let shifted = &output - &max_per_row.insert_axis(Axis(1));
            let exp_scores = shifted.mapv(|x| x.exp());
            let sum_exp = exp_scores.sum_axis(Axis(1));
            let probabilities = &exp_scores / &sum_exp.insert_axis(Axis(1));

            let epsilon = 1e-8;
            let _loss = -(&targets * &probabilities.mapv(|p| (p + epsilon).ln())).sum() / batch_len;

            // println!("average loss: {:.4}", loss);

            let loss_grad = (&probabilities - &targets) / batch_len;
            let gradients = model.backward(&loss_grad, &trace);
            model.apply_gradients(gradients, learning_rate);
        }
        learning_rate *= 0.99;
        println!(
            "epoch {} complete. learning_rate {:.5}",
            e + 1,
            learning_rate
        );
        training_images.shuffle(&mut rand::rng());
    }

    // Validation
    let mut misclassifications: usize = 0;
    let test_batch_size = 256;

    for batch in test_images.chunks(test_batch_size) {
        let batch_inputs = stack(
            Axis(0),
            &batch
                .iter()
                .map(|(data, _)| data.view())
                .collect::<Vec<_>>(),
        )?;

        let predictions = model.forward(&batch_inputs);

        for (i, row) in predictions.axis_iter(Axis(0)).enumerate() {
            let prediction = row.argmax()?;
            if prediction != batch[i].1 as usize {
                misclassifications += 1;
            }
        }
    }

    println!(
        "Error Rate: {:.2}%",
        (misclassifications as f64 / test_images.len() as f64) * 100.0
    );

    Ok(())
}
