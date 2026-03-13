# RustNet-MNIST

RustNet-MNIST is a from-scratch deep learning framework built entirely in Rust. It implements a feedforward neural network capable of training on and classifying the MNIST dataset. The project avoids massive machine learning frameworks like PyTorch or TensorFlow, instead relying on the `ndarray` crate to perform matrix operations and implementing the forward and backward passes (backpropagation) manually.

## Features

* **Sequential Model API**: An easy-to-use API for stacking layers, similar to Keras (`sequential.rs`).
* **Custom Layers**:
  * **Dense Layer**: Fully connected layers with random weight initialization (`layer/dense_layer.rs`).
  * **Activation Layer**: Supports ReLU, LeakyReLU, and Identity functions (`layer/activation_layer.rs`).
  * **Dropout Layer**: Vectorized dropout implementation for regularization (`layer/dropout_layer.rs`).
* **Custom Dataset Loader**: Directly parses the standard MNIST IDX file format into memory-efficient contiguous arrays without excessive allocations (`loader.rs`).
* **Loss Function**: Stable Softmax Cross-Entropy with logits (`loss.rs`).
* **Optimizer**: Stochastic Gradient Descent (SGD) with learning rate decay (`optimizer.rs`).
* **Terminal Visualizer**: Uses terminal colors to render MNIST digits directly in the console (`visualizer.rs`).

## Network Architecture

By default, the `main.rs` file configures a Multi-Layer Perceptron (MLP) with the following architecture:
1. **Input**: 784 nodes (28x28 flattened image)
2. **Hidden Layer 1**: Dense (512 nodes) -> LeakyReLU -> Dropout (p=0.2)
3. **Hidden Layer 2**: Dense (256 nodes) -> LeakyReLU -> Dropout (p=0.2)
4. **Output Layer**: Dense (10 nodes for digit classes 0-9)

The model trains for 20 epochs using batches of 128 samples. The SGD optimizer starts with a learning rate of 0.1, decaying by a factor of 0.99 every epoch.

## Project Structure

* `src/main.rs`: Entry point containing the model definition, training loop, and evaluation loop.
* `src/loader.rs`: MNIST dataset parser (`train-images-idx3-ubyte`, etc.).
* `src/sequential.rs`: The container that chains layers and manages the forward/backward pass cycles.
* `src/loss.rs`: Loss calculations and their gradients.
* `src/optimizer.rs`: Traits and implementations for weight updates.
* `src/visualizer.rs`: Utilities for viewing network inputs/outputs in the terminal.
* `src/layer/`: Module containing implementations of the network layers (`mod.rs`, `dense_layer.rs`, `activation_layer.rs`, `dropout_layer.rs`).

## Dependencies

The project relies on a few core Rust crates for math and random number generation. (You will need these in your `Cargo.toml`):
* `ndarray`
* `ndarray-stats`
* `ndarray-rand`
* `rand`
* `rand_distr`
* `colored`

## Getting Started

### 1. Download the Dataset
Create a folder named `dataset` in the root of the project and download the standard MNIST files into it. The required files are:
* `train-images-idx3-ubyte`
* `train-labels-idx1-ubyte`
* `t10k-images-idx3-ubyte`
* `t10k-labels-idx1-ubyte`

### 2. Run the Model
Make sure you have Rust and Cargo installed, then simply run:

```bash
cargo run --release
```

*(Note: Running with `--release` is highly recommended as matrix multiplication and backpropagation are computationally heavy and will run significantly faster compiled with optimizations).*

## Output
When you run the project, it will load the dataset, begin the training process over 20 epochs, print the decaying learning rate, and finally output the test error rate on the 10,000 holdout images.

Example Output:
```text
Training data loaded
epoch 1 complete. learning_rate 0.09900
epoch 2 complete. learning_rate 0.09801
...
epoch 20 complete. learning_rate 0.08179
Error Rate: 1.67%
```
