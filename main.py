import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sf = 2**20
max_int32 = 2**31 - 1

class FixedPointArray:
    def __init__(self, data):
        self.data = np.clip(np.round(data * sf), -max_int32, max_int32).astype(np.int32)

    @property
    def shape(self):
        return self.data.shape

def int_to_float(x):
    return x / sf

def int_add(a, b):
    return np.clip(np.add(a, b, dtype=np.int64), -max_int32, max_int32).astype(np.int32)

def int_sub(a, b):
    return np.clip(np.subtract(a, b, dtype=np.int64), -max_int32, max_int32).astype(np.int32)

def int_mul(a, b):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]:
        # Vectorized matrix multiplication
        result = np.dot(a, b) // sf
        return np.clip(result, -max_int32, max_int32).astype(np.int32)
    else:
        # Element-wise multiplication
        return np.clip((a * b) // sf, -max_int32, max_int32).astype(np.int32)

def int_div(a, b):
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    return np.clip((a * sf) // np.maximum(b, 1), -max_int32, max_int32).astype(np.int32)

def int_leaky_relu(x, alpha=0.01):
    alpha_int = int(alpha * sf)
    return np.where(x > 0, x, (x * alpha_int) // sf)

def int_softmax(x):
    # Use floating-point arithmetic for softmax
    x_float = x.astype(np.float64) / sf
    x_max = np.max(x_float, axis=-1, keepdims=True)
    exp_x = np.exp(x_float - x_max)
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    softmax = exp_x / sum_exp_x
    return (softmax * sf).astype(np.int32)

def int_cross_entropy_loss(y_true, y_pred):
    # Use floating-point arithmetic for loss computation
    y_true_float = y_true.astype(np.float64) / sf
    y_pred_float = y_pred.astype(np.float64) / sf
    epsilon = 1e-8
    y_pred_float = np.clip(y_pred_float, epsilon, 1.0 - epsilon)
    loss = -np.sum(y_true_float * np.log(y_pred_float)) / y_pred.shape[0]
    return loss

class IntLayer:
    def __init__(self, input_size, output_size, activation='leaky_relu'):
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.clip(np.round(np.random.uniform(-limit, limit, (input_size, output_size)) * sf), -max_int32, max_int32).astype(np.int32)
        self.biases = np.zeros(output_size, dtype=np.int32)
        self.activation = activation

    def forward(self, inputs):
        self.inputs = inputs
        self.z = int_add(int_mul(inputs, self.weights), self.biases)
        if self.activation == 'leaky_relu':
            output = int_leaky_relu(self.z)
        elif self.activation == 'softmax':
            output = int_softmax(self.z)
        else:
            output = self.z
        return output

    def backward(self, grad_output, learning_rate):
        if self.activation == 'leaky_relu':
            grad_output = np.where(self.z > 0, grad_output, int_mul(grad_output, int(0.01 * sf)))
        elif self.activation == 'softmax':
            pass  # Gradient is already computed in the output layer

        weight_grads = int_mul(self.inputs.T, grad_output)
        bias_grads = np.sum(grad_output, axis=0)

        # Gradient clipping
        max_grad = 1000 * sf
        weight_grads = np.clip(weight_grads, -max_grad, max_grad)
        bias_grads = np.clip(bias_grads, -max_grad, max_grad)

        lr_fixed = int(learning_rate * sf)
        # Update weights: weights -= learning_rate * weight_grads
        self.weights = int_sub(self.weights, int_mul(weight_grads, lr_fixed) // sf)
        self.biases = int_sub(self.biases, int_mul(bias_grads, lr_fixed) // sf)

        grad_input = int_mul(grad_output, self.weights.T)
        return grad_input

class IntNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            activation = 'leaky_relu' if i < len(layer_sizes) - 2 else 'softmax'
            self.layers.append(IntLayer(layer_sizes[i], layer_sizes[i+1], activation))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, x, y, learning_rate):
        outputs = self.forward(x)
        grad = int_sub(outputs, y)  # Gradient of softmax-crossentropy is outputs - y_true
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
        return grad

    def train(self, X, y, X_val, y_val, epochs, batch_size, learning_rate, patience=5):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()

            # Shuffle the training data
            indices = np.random.permutation(len(X))
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            total_batches = len(X) // batch_size

            for i in range(0, len(X), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]

                # Backward pass
                self.backward(batch_X, batch_y, learning_rate)

                # Print progress every 20 batches
                if (i // batch_size) % 20 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {i//batch_size + 1}/{total_batches}")

            # Evaluate on a small subset of the data every epoch
            train_subset = np.random.choice(len(X), size=500, replace=False)
            val_subset = np.random.choice(len(X_val), size=500, replace=False)

            train_outputs = self.forward(X[train_subset])
            train_loss = int_cross_entropy_loss(y[train_subset], train_outputs)
            train_acc = np.mean(np.argmax(int_to_float(train_outputs), axis=1) == np.argmax(int_to_float(y[train_subset]), axis=1))

            val_outputs = self.forward(X_val[val_subset])
            val_loss = int_cross_entropy_loss(y_val[val_subset], val_outputs)
            val_acc = np.mean(np.argmax(int_to_float(val_outputs), axis=1) == np.argmax(int_to_float(y_val[val_subset]), axis=1))

            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.2f}s")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    def predict(self, X):
        outputs = self.forward(X)
        return np.argmax(outputs, axis=1)

# Data loading and preprocessing
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
X = X.astype(np.float32)
y = y.astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = FixedPointArray(scaler.fit_transform(X_train)).data
X_val = FixedPointArray(scaler.transform(X_val)).data
X_test = FixedPointArray(scaler.transform(X_test)).data

def one_hot_encode(y, num_classes=10):
    return (np.eye(num_classes)[y] * sf).astype(np.int32)

y_train_onehot = one_hot_encode(y_train)
y_val_onehot = one_hot_encode(y_val)
y_test_onehot = one_hot_encode(y_test)

print("Data preprocessing completed.")

print("Creating and training the network...")
network = IntNetwork([784, 128, 64, 10])
network.train(X_train, y_train_onehot, X_val, y_val_onehot, epochs=100, batch_size=256, learning_rate=0.001, patience=5)
