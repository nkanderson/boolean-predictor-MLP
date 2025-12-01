#include "mlp.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @brief Parse a CSV line to extract target and input bits
 *
 * @param line CSV line in format: <target>,<64-bit number>
 * @param input_size Number of lowest bits to use as input
 * @return std::pair<float, std::vector<float>> Target and input vector
 */
std::pair<float, std::vector<float>> parse_csv_line(const std::string &line,
                                                    unsigned int input_size) {
  std::stringstream ss(line);
  std::string target_str, input_str;

  // Parse: target,input
  if (!std::getline(ss, target_str, ',') || !std::getline(ss, input_str)) {
    throw std::runtime_error("Invalid CSV format: " + line);
  }

  // Parse target (0 or 1)
  float target = std::stof(target_str);
  if (target != 0.0f && target != 1.0f) {
    throw std::runtime_error("Target must be 0 or 1, got: " + target_str);
  }

  // Parse input as 64-bit unsigned integer
  unsigned long long input_num = std::stoull(input_str);

  // Extract the lowest input_size bits as a vector of floats (0.0 or 1.0)
  std::vector<float> inputs(input_size);
  for (unsigned int i = 0; i < input_size; ++i) {
    inputs[i] = static_cast<float>((input_num >> i) & 1);
  }

  return {target, inputs};
}

/**
 * @brief Train MLP on CSV data in streaming/chunked fashion
 *
 * Reads CSV file in batches to avoid loading entire dataset into memory.
 * Tracks and reports loss during training for convergence monitoring.
 *
 * @param network MLP network to train
 * @param filename Path to CSV file
 * @param input_size Number of lowest bits to use as input
 * @param epochs Number of passes through the dataset
 * @param learning_rate Learning rate for gradient descent
 * @param batch_size Number of samples per training batch
 * @param loss_report_frequency Report loss every N epochs (0 = no reporting)
 * @return size_t Total number of samples processed
 */
size_t train_streaming(mlp::MLP &network, const std::string &filename,
                       unsigned int input_size, unsigned int epochs,
                       float learning_rate, size_t batch_size,
                       unsigned int loss_report_frequency = 10) {
  size_t total_samples = 0;
  std::vector<std::vector<float>> all_inputs;
  std::vector<float> all_targets;

  // Flag to track if we've loaded data for loss computation
  bool data_loaded_for_loss = false;

  for (unsigned int epoch = 0; epoch < epochs; ++epoch) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + filename);
    }

    std::vector<std::vector<float>> batch_inputs;
    std::vector<float> batch_targets;
    std::string line;
    size_t line_number = 0;
    size_t epoch_samples = 0;

    batch_inputs.reserve(batch_size);
    batch_targets.reserve(batch_size);

    while (std::getline(file, line)) {
      ++line_number;

      // Skip empty lines
      if (line.empty() ||
          line.find_first_not_of(" \t\r\n") == std::string::npos) {
        continue;
      }

      try {
        auto [target, inputs] = parse_csv_line(line, input_size);
        batch_targets.push_back(target);
        batch_inputs.push_back(inputs);
        ++epoch_samples;

        // Store data for loss computation (first epoch only)
        if (epoch == 0 && loss_report_frequency > 0 && !data_loaded_for_loss) {
          all_inputs.push_back(inputs);
          all_targets.push_back(target);
        }

        // Train when batch is full
        if (batch_inputs.size() >= batch_size) {
          network.train(batch_inputs, batch_targets, 1, learning_rate);
          batch_inputs.clear();
          batch_targets.clear();
        }

      } catch (const std::exception &e) {
        std::cerr << "Error on line " << line_number << ": " << e.what()
                  << std::endl;
        throw;
      }
    }

    // Train on any remaining samples in the last incomplete batch
    if (!batch_inputs.empty()) {
      network.train(batch_inputs, batch_targets, 1, learning_rate);
    }

    file.close();

    // Mark that we've loaded all data
    if (epoch == 0) {
      total_samples = epoch_samples;
      data_loaded_for_loss = true;
    }

    // Compute and report loss and accuracy
    bool should_report =
        (loss_report_frequency > 0) &&
        ((epoch % loss_report_frequency == 0) || (epoch == epochs - 1));

    if (should_report) {
      float loss = network.compute_loss(all_inputs, all_targets);
      float accuracy = network.compute_accuracy(all_inputs, all_targets);
      std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                << " - Loss: " << loss 
                << " - Accuracy: " << (accuracy * 100.0f) << "%"
                << std::endl;
    }
  }

  return total_samples;
}

void print_usage(const char *program_name) {
  std::cout << "Usage: " << program_name
            << " <csv_file> <input_size> <hidden_layer_size> [epochs] "
               "[learning_rate] [batch_size]\n";
  std::cout << "\n";
  std::cout << "Arguments:\n";
  std::cout << "  csv_file          - Path to CSV training data file\n";
  std::cout << "                      Format: <target>,<64-bit number>\n";
  std::cout << "  input_size        - Number of lowest bits to use as input "
               "(1-64)\n";
  std::cout << "  hidden_layer_size - Number of hidden layer neurons\n";
  std::cout << "  epochs            - Number of training epochs (default: "
               "1000)\n";
  std::cout << "  learning_rate     - Learning rate (default: 0.1)\n";
  std::cout << "  batch_size        - Samples per batch (default: 32)\n";
  std::cout << "\n";
  std::cout << "Example:\n";
  std::cout << "  " << program_name << " training_data.csv 16 8 5000 0.5 64\n";
  std::cout << "\n";
  std::cout << "Note: Uses streaming/chunked training for large datasets.\n";
}

int main(int argc, char *argv[]) {
  // Parse command line arguments
  if (argc < 4 || argc > 7) {
    print_usage(argv[0]);
    return 1;
  }

  std::string csv_file = argv[1];
  unsigned int input_size = std::stoul(argv[2]);
  unsigned int hidden_layer_size = std::stoul(argv[3]);
  unsigned int epochs = (argc >= 5) ? std::stoul(argv[4]) : 1000;
  float learning_rate = (argc >= 6) ? std::stof(argv[5]) : 0.1f;
  size_t batch_size = (argc >= 7) ? std::stoul(argv[6]) : 32;

  // Validate input_size
  if (input_size == 0 || input_size > 64) {
    std::cerr << "Error: input_size must be between 1 and 64\n";
    return 1;
  }

  // Validate batch_size
  if (batch_size == 0) {
    std::cerr << "Error: batch_size must be at least 1\n";
    return 1;
  }

  try {
    // Create MLP
    std::cout << "Creating MLP with:\n";
    std::cout << "  Input size: " << input_size << std::endl;
    std::cout << "  Hidden layer size: " << hidden_layer_size << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    mlp::MLP network(input_size, hidden_layer_size);

    // Train the network with streaming
    std::cout << "\nTraining from: " << csv_file << std::endl;
    std::cout << "Epochs: " << epochs << ", Learning rate: " << learning_rate
              << std::endl;
    std::cout << "\nStarting training...\n";

    // Report loss every 10 epochs or at least 10 times total
    unsigned int loss_report_freq = std::max(1u, epochs / 10);

    size_t total_samples =
        train_streaming(network, csv_file, input_size, epochs, learning_rate,
                        batch_size, loss_report_freq);

    std::cout << "\nTraining complete!" << std::endl;
    std::cout << "Total samples per epoch: " << total_samples << std::endl;

    // Save weights
    std::cout << "\nSaving weights..." << std::endl;
    network.save_weights();
    std::string weights_file = "mlp_" + std::to_string(input_size) + "_" +
                               std::to_string(hidden_layer_size) + ".txt";
    std::cout << "Weights saved to: " << weights_file << std::endl;

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
