# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -O2
DEBUG_FLAGS = -g -O0

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
LIB_DIR = lib
BIN_DIR = bin

# Library name
LIB_NAME = mlp
STATIC_LIB = $(LIB_DIR)/lib$(LIB_NAME).a
SHARED_LIB = $(LIB_DIR)/lib$(LIB_NAME).so

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))

# Example executable
EXAMPLE_SRC = example.cpp
EXAMPLE_BIN = $(BIN_DIR)/example

# Default target
.PHONY: all
all: directories static

# Create necessary directories
.PHONY: directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(LIB_DIR)
	@mkdir -p $(BIN_DIR)

# Build static library
.PHONY: static
static: $(STATIC_LIB)

$(STATIC_LIB): $(OBJECTS)
	ar rcs $@ $^
	@echo "Static library created: $@"

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Build example executable
.PHONY: example
example: directories static
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) $(EXAMPLE_SRC) -L$(LIB_DIR) -l$(LIB_NAME) -o $(EXAMPLE_BIN)
	@echo "Example executable created: $(EXAMPLE_BIN)"

# Run the example
.PHONY: run-example
run-example: example
	@echo "Running example..."
	@$(EXAMPLE_BIN)

# Debug build
.PHONY: debug
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: clean all

# Clean build files
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR) $(BIN_DIR)
	@echo "Cleaned build files"

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all         - Build static library (default)"
	@echo "  static      - Build static library"
	@echo "  example     - Build example executable"
	@echo "  run-example - Build and run the example"
	@echo "  debug       - Build with debug symbols"
	@echo "  clean       - Remove build artifacts"
	@echo "  help        - Show this help message"
