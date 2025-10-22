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

  // Example 4: Forward propagation
  std::cout << "Example 4: Forward propagation with network3" << std::endl;

  std::vector<float> test_input1 = {0.0f, 0.0f};
  float output1 = network3.forward(test_input1);
  std::cout << "  Input: [" << test_input1[0] << ", " << test_input1[1]
            << "] -> Output: " << output1 << std::endl;

  std::vector<float> test_input2 = {1.0f, 0.0f};
  float output2 = network3.forward(test_input2);
  std::cout << "  Input: [" << test_input2[0] << ", " << test_input2[1]
            << "] -> Output: " << output2 << std::endl;

  std::vector<float> test_input3 = {0.0f, 1.0f};
  float output3 = network3.forward(test_input3);
  std::cout << "  Input: [" << test_input3[0] << ", " << test_input3[1]
            << "] -> Output: " << output3 << std::endl;

  std::vector<float> test_input4 = {1.0f, 1.0f};
  float output4 = network3.forward(test_input4);
  std::cout << "  Input: [" << test_input4[0] << ", " << test_input4[1]
            << "] -> Output: " << output4 << std::endl;
  std::cout << std::endl;

  // Example 5: Training a network to learn XOR function
  std::cout << "Example 5: Training a network to learn XOR" << std::endl;

  // Create network with random weights
  mlp::MLP xor_network(2, 4); // 2 inputs, 4 hidden neurons

  // XOR training data
  std::vector<std::vector<float>> xor_inputs = {
      {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
  std::vector<float> xor_targets = {0.0f, 1.0f, 1.0f, 0.0f};

  std::cout << "  Before training:" << std::endl;
  for (size_t i = 0; i < xor_inputs.size(); ++i) {
    float output = xor_network.forward(xor_inputs[i]);
    std::cout << "    [" << xor_inputs[i][0] << ", " << xor_inputs[i][1]
              << "] -> " << output << " (target: " << xor_targets[i] << ")"
              << std::endl;
  }

  // Train the network
  std::cout << "  Training for 5000 epochs..." << std::endl;
  xor_network.train(xor_inputs, xor_targets, 5000, 0.5f);

  std::cout << "  After training:" << std::endl;
  for (size_t i = 0; i < xor_inputs.size(); ++i) {
    float output = xor_network.forward(xor_inputs[i]);
    std::cout << "    [" << xor_inputs[i][0] << ", " << xor_inputs[i][1]
              << "] -> " << output << " (target: " << xor_targets[i] << ")"
              << std::endl;
  }
  std::cout << std::endl;

  std::cout << "=== All examples completed successfully ===" << std::endl;

  return 0;
}
