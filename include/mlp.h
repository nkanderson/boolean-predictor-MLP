#ifndef MLP_H
#define MLP_H

#include <ostream>
#include <vector>

namespace mlp {

/**
 * @brief Multi-Layer Perceptron class
 *
 * A basic implementation of a multi-layer perceptron neural network
 * for boolean prediction.
 */
class MLP {
public:
  /**
   * @brief Construct a new MLP object
   *
   * @param input_size Number of input neurons
   * @param hidden_layer_size Number of neurons in the hidden layer (default: 2)
   * @param hidden_weights Weights and biases for the hidden layer
   * (Input→Hidden). Each inner vector contains weights and bias for one hidden
   * neuron (input_size + 1 (for bias) elements each).
   * @param output_weights Weights and bias for the output layer
   * (Hidden→Output). Single vector with hidden_layer_size + 1 elements.
   */
  explicit MLP(unsigned int input_size, unsigned int hidden_layer_size = 2,
               const std::vector<std::vector<float>> &hidden_weights =
                   std::vector<std::vector<float>>(),
               const std::vector<float> &output_weights = std::vector<float>());

  /**
   * @brief Destroy the MLP object
   */
  ~MLP();

  /**
   * @brief Stream insertion operator for printing MLP
   */
  friend std::ostream &operator<<(std::ostream &os, const MLP &mlp);

private:
  /**
   * @brief Generate random weights
   *
   * @param size Number of weights to generate
   * @return std::vector<float> Vector of random weights in range [-1.0, 1.0]
   */
  static std::vector<float> generate_random_weights(size_t size);

  unsigned int input_size_;
  unsigned int hidden_layer_size_;
  std::vector<std::vector<float>> hidden_weights_; // Input→Hidden
  std::vector<float> output_weights_;              // Hidden→Output
};

} // namespace mlp

#endif // MLP_H
