#include "mlp.h"
#include <random>
#include <stdexcept>

namespace mlp {

MLP::MLP(unsigned int input_size, unsigned int hidden_layer_size,
         const std::vector<float> &input_weights,
         const std::vector<std::vector<float>> &hidden_weights)
    : input_size_(input_size), hidden_layer_size_(hidden_layer_size),
      input_weights_(input_weights), hidden_weights_(hidden_weights) {

  // Initialize or validate input_weights
  // Expected size: input_size + 1 (for bias)
  const size_t expected_input_weights_size = input_size_ + 1;

  if (input_weights_.empty()) {
    // Initialize with random weights
    std::random_device rd;  // True random seed using hardware entropy
    std::mt19937 gen(rd()); // Fast PRNG (Mersenne Twister)
    std::uniform_real_distribution<float> dist(-1.0f,
                                               1.0f); // Range: [-1.0, 1.0]

    input_weights_.resize(expected_input_weights_size);
    for (size_t i = 0; i < expected_input_weights_size; ++i) {
      input_weights_[i] = dist(gen); // Generate random weight
    }
  } else {
    // Validate size
    if (input_weights_.size() != expected_input_weights_size) {
      throw std::invalid_argument("input_weights size mismatch: expected " +
                                  std::to_string(expected_input_weights_size) +
                                  " but got " +
                                  std::to_string(input_weights_.size()));
    }
  }

  // TODO: Initialize random weights and biases if hidden_weights is empty
  // TODO: Validate hidden_weights structure matches expected dimensions
}

MLP::~MLP() {
  // Cleanup if needed
}

std::ostream &operator<<(std::ostream &os, const MLP &mlp) {
  os << "MLP(\n";
  os << "  input_size: " << mlp.input_size_ << "\n";
  os << "  hidden_layer_size: " << mlp.hidden_layer_size_ << "\n";

  // Print input weights
  os << "  input_weights: [";
  for (size_t i = 0; i < mlp.input_weights_.size(); ++i) {
    os << mlp.input_weights_[i];
    if (i < mlp.input_weights_.size() - 1) {
      os << ", ";
    }
  }
  os << "]\n";

  // Print hidden weights
  os << "  hidden_weights: [\n";
  for (size_t i = 0; i < mlp.hidden_weights_.size(); ++i) {
    os << "    neuron " << i << ": [";
    for (size_t j = 0; j < mlp.hidden_weights_[i].size(); ++j) {
      os << mlp.hidden_weights_[i][j];
      if (j < mlp.hidden_weights_[i].size() - 1) {
        os << ", ";
      }
    }
    os << "]";
    if (i < mlp.hidden_weights_.size() - 1) {
      os << ",";
    }
    os << "\n";
  }
  os << "  ]\n";
  os << ")";

  return os;
}

} // namespace mlp
