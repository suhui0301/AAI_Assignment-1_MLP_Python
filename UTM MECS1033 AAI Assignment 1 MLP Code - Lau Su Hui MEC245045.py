import numpy as np

np.random.seed(1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias_output = np.zeros((1, output_size))

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y, output):
        error = y - output
        d_output = error * sigmoid_derivative(output)
        error_hidden = d_output.dot(self.weights_hidden_output.T)
        d_hidden = error_hidden * sigmoid_derivative(self.hidden_output)

        self.weights_hidden_output += self.hidden_output.T.dot(d_output) * self.learning_rate
        self.bias_output += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate

        self.weights_input_hidden += X.T.dot(d_hidden) * self.learning_rate
        self.bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs):
        print(f"Starting training process for {epochs} cycles...")

        for i in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if (i % 1000) == 0:
                loss = np.mean(np.square(y - output))  # How wrong is the model?
                print(f"Epoch {i}: Loss {loss:.6f}")  # We want Loss to go to 0

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    print("--- 1. INITIALIZATION ---")
    nn = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)

    print("\n--- 2. PRE-TRAINING CHECK ---")
    print("Predictions before learning (Should be random):")
    print(nn.forward(X))

    print("\n--- 3. TRAINING PHASE ---")
    nn.train(X, y, epochs=10000)

    print("\n--- 4. FINAL RESULTS ---")
    final_output = nn.forward(X)

    print(f"{'Input':<15} {'Expected':<10} {'Predicted':<15}")
    print("-" * 45)
    for i in range(len(X)):
        # We round the random numbers to make them readable strings
        pred_val = f"{final_output[i][0]:.4f}"
        print(f"{str(X[i]):<15} {str(y[i]):<10} {pred_val:<15}")
        
