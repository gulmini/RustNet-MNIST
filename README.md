# Rust MNIST Neural Network

A lightweight, from-scratch deep learning framework built in Rust. This project implements a fully functional Feedforward Neural Network (Multilayer Perceptron) to classify handwritten digits from the MNIST dataset. It uses the `ndarray` crate for efficient matrix operations and avoids heavy machine learning dependencies like PyTorch or TensorFlow.

## Features

* **Custom Neural Network API**: A `Sequential` model builder that allows chaining various layers together.
* **Layer Types**:
  * **Dense (Fully Connected) Layer**: Supports He initialization for robust training.
  * **Dropout Layer**: Implements inverted dropout for regularization to prevent overfitting.
* **Activation Functions**: Includes `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`, and `Identity`.
* **Custom Dataset Loader**: Parses the raw IDX file format used by the MNIST dataset natively.
* **Training Pipeline**: Implements Stochastic Gradient Descent (SGD) with mini-batching, learning rate decay, and a Softmax + Cross-Entropy loss function computed from scratch.
* **Terminal Visualization**: Includes a visualizer module to render images directly in the console using colored terminal blocks.

## Project Structure

```text
src/
├── main.rs                 # Entry point: model definition, training loop, and evaluation
├── loader.rs               # MNIST IDX format parser and data loader
├── sequential.rs           # Sequential model handling forward and backward passes
├── visualizer.rs           # Terminal-based image visualization utilities
└── layer/
    ├── mod.rs              # Layer traits, LayerGradients, and Activation functions
    ├── dense_layer.rs      # Fully connected layer implementation
    └── dropout_layer.rs    # Dropout regularization layer implementation
```

Model Architecture

By default, main.rs configures the following architecture to process 28x28 grayscale images (784 flattened pixels):

1. Input: 784 neurons
2. Dense Layer: 512 neurons (LeakyReLU activation)
3. Dropout Layer: 20% drop probability
4. Dense Layer: 256 neurons (LeakyReLU activation)
5. Dropout Layer: 20% drop probability
6. Output Dense Layer: 10 neurons (Identity activation)
7. Loss: Cross-Entropy Loss with Softmax probabilities

Dependencies

This project relies on the following Rust crates:
* ndarray: For N-dimensional arrays and fast matrix multiplications.
* ndarray-stats: For array statistical operations.
* ndarray-rand / rand: For weight initialization and dropout mask generation.
* colored: For terminal output formatting in the visualizer.


## Getting Started
1. Download the Dataset

Download the standard MNIST dataset files and place them in a dataset/ directory at the root of your project:
- train-images-idx3-ubyte
- train-labels-idx1-ubyte
- t10k-images-idx3-ubyte
- t10k-labels-idx1-ubyte

2. Run the Model

Execute the training loop and evaluation using Cargo:
```bash
cargo run --release
```

*Note: Running with --release is highly recommended for ndarray matrix operations, as debug mode will be significantly slower).
Output*

The training process will output the progress for 30 epochs, automatically applying learning rate decay. After training, it will run a forward pass on the 10,000 test images and output the final Error Rate percentage.
