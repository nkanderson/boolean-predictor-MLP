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
   * @param input_weights Weights and bias for the input layer
   * @param hidden_weights Weights and biases for the hidden layer.
   *                       Each inner vector contains weights and bias for one
   *                       hidden neuron.
   */
  explicit MLP(unsigned int input_size, unsigned int hidden_layer_size = 2,
               const std::vector<float> &input_weights = std::vector<float>(),
               const std::vector<std::vector<float>> &hidden_weights =
                   std::vector<std::vector<float>>());

  /**
   * @brief Destroy the MLP object
   */
  ~MLP();

  /**
   * @brief Stream insertion operator for printing MLP
   */
  friend std::ostream &operator<<(std::ostream &os, const MLP &mlp);

private:
  unsigned int input_size_;
  unsigned int hidden_layer_size_;
  std::vector<float> input_weights_;
  std::vector<std::vector<float>> hidden_weights_;
};

} // namespace mlp

#endif // MLP_H
