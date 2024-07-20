# Float2Int: An Efficient Neural Network with Integer Operations

## Introduction

Float2Int is a neural network designed to leverage integer arithmetic for improved computational efficiency without compromising performance. Traditional neural networks rely on floating-point operations, which can be computationally expensive. By converting floating-point numbers to integers, Float2Int aims to reduce the computational load, leading to faster training and inference times. This approach is conceptually similar to quantization, where neural networks maintain high accuracy even with lower precision representations.

## Methodology

### Integer Representation

Float2Int uses a simple but effective method to represent floating-point numbers as integers. Given a floating-point number \( x \), it is scaled by a factor of \( 10^6 \) (or any other suitable scale) and then converted to an integer. For example:
\[ x_{\text{int}} = \text{round}(x \times 10^6) \]

To convert back to the original floating-point representation, the integer is scaled down:
\[ x_{\text{float}} = x_{\text{int}} \times 10^{-6} \]

### Mathematical Operations

Key mathematical operations are implemented to work directly with these scaled integers. This includes:

1. **Matrix Multiplication**:
\[ C[i, j] = \left( \sum_{k} A[i, k] \times B[k, j] \right) // \text{SCALE} \]

2. **ReLU Activation**:
\[ \text{ReLU}(x) = \max(0, x) \]

3. **Exponential Function (Taylor Series Approximation)**:
\[ \text{exp}(x) \approx \text{SCALE} + x + \frac{x^2}{2 \times \text{SCALE}} + \frac{x^3}{6 \times \text{SCALE}^2} \]

4. **Softmax Function**:
\[ \text{softmax}(x_i) = \frac{\text{exp}(x_i)}{\sum_j \text{exp}(x_j)} \]

5. **Logarithm (Taylor Series Approximation)**:
\[ \log(1 + x) \approx x - \frac{x^2}{2} + \frac{x^3}{3} \]

### Neural Network Architecture

The network is constructed using layers of neurons, where each layer performs integer-based operations. The architecture typically includes:

- Input layer: Takes the scaled integer inputs.
- Hidden layers: Perform matrix multiplications, followed by ReLU activations.
- Output layer: Computes the softmax probabilities.

### Training

During training, the network performs forward and backward passes using integer arithmetic. The loss function, cross-entropy in this case, is computed in the integer domain. Gradients are also represented as integers, and the weights are updated using integer arithmetic.

## Results

*Results will be populated here once the experiments are conducted.*
