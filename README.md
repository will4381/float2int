# float2int: Fixed-Point Neural Network Implementation

## Overview

float2int is an innovative neural network architecture that uses fixed-point arithmetic for improved performance and efficiency. This implementation demonstrates how to convert floating-point operations to integer operations in neural networks, potentially leading to faster computation and reduced memory usage.

## Note on Implementation

This is a basic implementation of the float2int architecture. Due to time constraints, some optimizations and advanced features are not yet implemented. However, this serves as a solid foundation for further development and exploration of fixed-point arithmetic in neural networks.

I encourage contributions from the community! If you have ideas for improvements or want to extend the functionality, please feel free to fork the repository and submit pull requests. Your input can help evolve this project into a more robust and efficient framework for fixed-point neural networks.

## Introduction

float2int converts standard floating-point neural network operations to fixed-point integer operations. This approach can offer several advantages:

- Reduced memory usage
- Potential for faster computation
- Improved compatibility with hardware accelerators

The implementation includes a full neural network pipeline, from data preprocessing to training and evaluation, all using fixed-point arithmetic.

## Architecture

float2int implements a multi-layer perceptron (MLP) using fixed-point arithmetic. Key components include:

1. `FixedPointArray`: Converts floating-point data to fixed-point representation
2. `IntLayer`: Implements a neural network layer using integer operations
3. `IntNetwork`: Combines multiple IntLayers to form a complete neural network

The architecture supports various activation functions (ReLU, softmax) and implements backpropagation for training.

## Mathematical Foundation

### Scale Factor

The core of float2int is the scale factor ($`sf`$), which converts floating-point numbers to fixed-point integers:

$`x_{int} = round(x_{float} \cdot sf)`$

We use $`sf = 2^{20}`$ in this implementation. This choice balances precision and the range of representable values.

### Integer Operations

Basic arithmetic operations are modified to work with fixed-point integers:

1. Addition and Subtraction: Performed directly on integers
2. Multiplication: $`z = \frac{x \cdot y}{sf}`$
3. Division: $`z = \frac{x \cdot sf}{y}`$

### Activation Functions

ReLU remains straightforward:

$`ReLU(x) = max(0, x)`$

Softmax requires more careful handling:

$`softmax(x_i) = \frac{e^{x_i / sf}}{\sum_j e^{x_j / sf}} \cdot sf`$

### Loss Function

The cross-entropy loss is adapted for fixed-point:

$`L = -\frac{1}{N} \sum_i y_i \cdot log(\frac{y_{pred,i}}{sf})`$

### Backpropagation

Gradients are computed and applied using fixed-point arithmetic, maintaining the scale factor throughout the process.

## Performance

The performance of float2int is demonstrated on the MNIST dataset. The implementation includes data loading, preprocessing, training, and evaluation.

## Results

[This section intentionally left blank for you to fill in with actual results from running the model]

## Further Improvements

[This section intentionally left blank for you to suggest potential enhancements to the architecture]

## Citation

If you use float2int in your research, please cite:

```
Will Kusch. (2024). float2int: Fixed-Point Neural Network Implementation. Relative Companies.
```

For further information, contact me at will@relativecompanies.com.
