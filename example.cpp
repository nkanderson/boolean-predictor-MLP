#include "mlp.h"
#include <iostream>
#include <vector>

int main() {
  std::cout << "=== MLP Boolean Predictor Example ===" << std::endl;
  std::cout << std::endl;

  // Example 1: Create MLP with default hidden layer size and no weights
  std::cout << "Example 1: Basic MLP (2 inputs, 2 hidden neurons)" << std::endl;
  mlp::MLP network1(2); // 2 inputs, default 2 hidden neurons
  std::cout << network1 << std::endl;
  std::cout << std::endl;

  // Example 2: Create MLP with custom hidden layer size
  std::cout
      << "Example 2: MLP with custom hidden layer (2 inputs, 4 hidden neurons)"
      << std::endl;
  mlp::MLP network2(2, 4); // 2 inputs, 4 hidden neurons
  std::cout << network2 << std::endl;
  std::cout << std::endl;

  // Example 3: Create MLP with predefined weights
  std::cout << "Example 3: MLP with predefined weights" << std::endl;

  // Hidden layer weights (Input→Hidden)
  // 2 hidden neurons, each with 2 input weights + 1 bias = 3 values
  std::vector<std::vector<float>> hidden_weights = {
      {0.8f, -0.2f, 0.4f}, // Weights for hidden neuron 0
      {-0.6f, 0.9f, -0.1f} // Weights for hidden neuron 1
  };

  // Output layer weights (Hidden→Output)
  // 1 output neuron with 2 hidden inputs + 1 bias = 3 values
  std::vector<float> output_weights = {0.5f, -0.3f, 0.1f};

  mlp::MLP network3(2, 2, hidden_weights, output_weights);
  std::cout << network3 << std::endl;
  std::cout << std::endl;

  std::cout << "=== All examples completed successfully ===" << std::endl;

  return 0;
}
