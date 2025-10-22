#include "mlp.h"
#include <random>
#include <stdexcept>

namespace mlp {

std::vector<float> MLP::generate_random_weights(size_t size) {
  std::random_device rd;  // True random seed
  std::mt19937 gen(rd()); // Fast PRNG (Mersenne Twister)
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f); // Range: [-1.0, 1.0]

  std::vector<float> weights(size);
  for (size_t i = 0; i < size; ++i) {
    weights[i] = dist(gen);
  }
  return weights;
}

MLP::MLP(unsigned int input_size, unsigned int hidden_layer_size,
         const std::vector<std::vector<float>> &hidden_weights,
         const std::vector<float> &output_weights)
    : input_size_(input_size), hidden_layer_size_(hidden_layer_size),
      hidden_weights_(hidden_weights), output_weights_(output_weights) {

  // Initialize or validate hidden_weights (Input→Hidden)
  // Expected: hidden_layer_size vectors, each with input_size + 1 (for bias)
  // elements
  const size_t expected_weights_per_hidden_neuron = input_size_ + 1;

  if (hidden_weights_.empty()) {
    // Initialize with random weights for each hidden neuron
    hidden_weights_.resize(hidden_layer_size_);
    for (size_t i = 0; i < hidden_layer_size_; ++i) {
      hidden_weights_[i] =
          generate_random_weights(expected_weights_per_hidden_neuron);
    }
  } else {
    // Validate structure
    if (hidden_weights_.size() != hidden_layer_size_) {
      throw std::invalid_argument("hidden_weights size mismatch: expected " +
                                  std::to_string(hidden_layer_size_) +
                                  " neurons but got " +
                                  std::to_string(hidden_weights_.size()));
    }
    // Validate each neuron's weights
    for (size_t i = 0; i < hidden_weights_.size(); ++i) {
      if (hidden_weights_[i].size() != expected_weights_per_hidden_neuron) {
        throw std::invalid_argument(
            "hidden_weights[" + std::to_string(i) +
            "] size mismatch: expected " +
            std::to_string(expected_weights_per_hidden_neuron) + " but got " +
            std::to_string(hidden_weights_[i].size()));
      }
    }
  }

  // Initialize or validate output_weights (Hidden→Output)
  // Expected size: hidden_layer_size + 1 (for bias)
  const size_t expected_output_weights_size = hidden_layer_size_ + 1;

  if (output_weights_.empty()) {
    // Initialize with random weights
    output_weights_ = generate_random_weights(expected_output_weights_size);
  } else {
    // Validate size
    if (output_weights_.size() != expected_output_weights_size) {
      throw std::invalid_argument("output_weights size mismatch: expected " +
                                  std::to_string(expected_output_weights_size) +
                                  " but got " +
                                  std::to_string(output_weights_.size()));
    }
  }
}

MLP::~MLP() {
  // Cleanup if needed
}

std::ostream &operator<<(std::ostream &os, const MLP &mlp) {
  os << "MLP(\n";
  os << "  input_size: " << mlp.input_size_ << "\n";
  os << "  hidden_layer_size: " << mlp.hidden_layer_size_ << "\n";

  // Print hidden weights (Input→Hidden)
  os << "  hidden_weights (Input→Hidden): [\n";
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

  // Print output weights (Hidden→Output)
  os << "  output_weights (Hidden→Output): [";
  for (size_t i = 0; i < mlp.output_weights_.size(); ++i) {
    os << mlp.output_weights_[i];
    if (i < mlp.output_weights_.size() - 1) {
      os << ", ";
    }
  }
  os << "]\n";
  os << ")";

  return os;
}

} // namespace mlp
