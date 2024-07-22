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
The core of float2int is the scale factor ($sf$), which converts floating-point numbers to fixed-point integers:
$x_{int} = round(x_{float} \cdot sf)$

We use $sf = 2^{8}$ in this implementation. This choice balances precision and the range of representable values.

### Integer Operations
Basic arithmetic operations are modified to work with fixed-point integers:
1. Addition and Subtraction: Performed directly on integers
2. Multiplication: $z = (x \cdot y) // sf$
3. Division: $z = (x \cdot sf) // y$

### Activation Functions
ReLU remains straightforward:
$ReLU(x) = max(0, x)$

Softmax requires more careful handling:
$softmax(x_i) = (e^{x_i / sf} \cdot sf) // \sum_j e^{x_j / sf}$

### Loss Function
The cross-entropy loss is adapted for fixed-point:
$L = -\sum_i y_i \cdot (log(y_{pred,i}) - log(sf)) // (N \cdot sf)$

### Backpropagation
Gradients are computed and applied using fixed-point arithmetic, maintaining the scale factor throughout the process.

## Performance
The performance of float2int is demonstrated on the MNIST dataset. The implementation includes data loading, preprocessing, training, and evaluation.

## Results
The current implementation shows limited learning capacity. The model achieves an accuracy of around 10-11% on both training and validation sets, which is only slightly better than random guessing for the 10-class MNIST dataset. The loss remains constant at about 0.0010, indicating that the model is not improving its predictions significantly during training.

## Current Problems
1. Slow Convergence: The model is not showing significant improvement over multiple epochs.
2. Low Accuracy: The model's performance is only slightly better than random guessing.
3. Constant Loss: The loss value remains unchanged, suggesting issues with gradient propagation or numerical stability.
4. Computational Efficiency: The training process is still slow, with matrix multiplications being a major bottleneck.
5. Numerical Precision: The current fixed-point representation may not provide enough precision for effective learning.

## Further Improvements
1. Implement more efficient matrix multiplication algorithms optimized for fixed-point arithmetic.
2. Experiment with different scale factors to find an optimal balance between precision and range.
3. Introduce layer-specific learning rates or adaptive learning rate methods.
4. Implement batch normalization using fixed-point arithmetic to help with training stability.
5. Explore quantization-aware training techniques to better handle the limitations of fixed-point representation.
6. Optimize the backpropagation algorithm for fixed-point arithmetic.
7. Implement more advanced optimization algorithms like Adam or RMSprop using fixed-point arithmetic.
8. Develop a custom CUDA kernel for fixed-point operations to leverage GPU acceleration.
9. Implement pruning and quantization techniques to further reduce model size and computation time.
10. Explore mixed-precision training, using higher precision for sensitive operations and lower precision for others.

## Citation
If you use float2int in your research, please cite:
```
Will Kusch. (2024). float2int: Fixed-Point Neural Network Implementation. Relative Companies.
```
For further information, contact me at will@relativecompanies.com.
