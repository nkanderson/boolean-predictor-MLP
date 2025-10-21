#include "mlp.h"
#include <iostream>
#include <vector>

int main() {
  std::cout << "=== MLP Boolean Predictor Example ===" << std::endl;
  std::cout << std::endl;

  // Example 1: Create MLP with default hidden layer size and no weights
  std::cout << "Example 1: Basic MLP (2 inputs, 2 hidden neurons)" << std::endl;
  mlp::MLP network1(2); // 2 inputs, default 2 hidden neurons
  std::cout << "  Created network with 2 inputs and 2 hidden neurons"
            << std::endl;
  std::cout << std::endl;

  // Example 2: Create MLP with custom hidden layer size
  std::cout
      << "Example 2: MLP with custom hidden layer (2 inputs, 4 hidden neurons)"
      << std::endl;
  mlp::MLP network2(2, 4); // 2 inputs, 4 hidden neurons
  std::cout << "  Created network with 2 inputs and 4 hidden neurons"
            << std::endl;
  std::cout << std::endl;

  // Example 3: Create MLP with predefined weights
  std::cout << "Example 3: MLP with predefined weights" << std::endl;

  // Input layer weights (for 2 inputs + 1 bias = 3 values)
  std::vector<float> input_weights = {0.5f, -0.3f, 0.1f};

  // Hidden layer weights (2 neurons, each with 2 inputs + 1 bias = 3 values)
  std::vector<std::vector<float>> hidden_weights = {
      {0.8f, -0.2f, 0.4f}, // Weights for hidden neuron 1
      {-0.6f, 0.9f, -0.1f} // Weights for hidden neuron 2
  };

  mlp::MLP network3(2, 2, input_weights, hidden_weights);
  std::cout << "  Created network with 2 inputs and 2 hidden neurons"
            << std::endl;
  std::cout << "  Input weights: [" << input_weights[0] << ", "
            << input_weights[1] << ", " << input_weights[2] << "]" << std::endl;
  std::cout << "  Hidden neuron 1 weights: [" << hidden_weights[0][0] << ", "
            << hidden_weights[0][1] << ", " << hidden_weights[0][2] << "]"
            << std::endl;
  std::cout << "  Hidden neuron 2 weights: [" << hidden_weights[1][0] << ", "
            << hidden_weights[1][1] << ", " << hidden_weights[1][2] << "]"
            << std::endl;
  std::cout << std::endl;

  std::cout << "=== All examples completed successfully ===" << std::endl;

  return 0;
}
