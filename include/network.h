#pragma once
#include "matrix.h"
#include "func.h"
#include "include.h"

/**
 * @struct data_network
 * @brief Structure to hold network configuration data.
 * 
 * This structure is used to store the configuration data for a neural network.
 */
struct data_network {
    /**
     * @brief Number of layers in the network.
     * 
     * Stores the number of layers in the network.
     */
    int L;

    /**
     * @brief Vector of layer sizes.
     * 
     * Stores the size of each layer in the network.
     */
    std::vector<int> size;
};

/**
 * @class Network
 * @brief Class to represent a neural network.
 * 
 * This class provides an interface for creating and manipulating neural networks.
 * It allows you to create a network with a specified configuration, set input values,
 * perform forward feed, backpropagation, and weight updates, and save and read weights.
 */
class Network {
private:
    /**
     * @brief Number of layers in the network.
     * 
     * Stores the number of layers in the network.
     */
    int L;

    /**
     * @brief Vector of layer sizes.
     * 
     * Stores the size of each layer in the network.
     */
    std::vector<int> size;

public:
    /**
     * @brief Activation function.
     * 
     * Stores the activation function used in the network.
     */
    Func f;

    /**
     * @brief Vector of weight matrices.
     * 
     * Stores the weight matrices for each layer in the network.
     */
    std::vector<Matrix> weights;

    /**
     * @brief Vector of bias values.
     * 
     * Stores the bias values for each layer in the network.
     */
    std::vector<std::vector<double>> bios;

    /**
     * @brief Vector of neuron values.
     * 
     * Stores the values of each neuron in the network.
     */
    std::vector<std::vector<double>> neurons_val;

    /**
     * @brief Vector of neuron error values.
     * 
     * Stores the error values of each neuron in the network.
     */
    std::vector<std::vector<double>> neurons_err;

    /**
     * @brief Vector of neuron bias values.
     * 
     * Stores the bias values of each neuron in the network.
     */
    std::vector<double> neurons_bios_val;

    /**
     * @brief Default constructor.
     * 
     * Initializes the network with default values.
     */
    Network() : L(0), size() {}

    /**
     * @brief Constructor with configuration data.
     * 
     * Initializes the network with the specified configuration data.
     * 
     * @param data Configuration data for the network.
     */
    Network(const data_network& data);

    /**
     * @brief Prints the network configuration.
     * 
     * Outputs the network configuration to the console.
     */
    void PrintConfig();

    /**
     * @brief Sets the input values for the network.
     * 
     * Sets the input values for the network.
     * 
     * @param values Input values for the network.
     */
    void SetInput(const std::vector<double>& values);

    /**
     * @brief Performs forward feed.
     * 
     * Performs forward feed through the network.
     * 
     * @return The output value of the network.
     */
    double ForwardFeed();

    /**
     * @brief Searches for the maximum index in a vector.
     * 
     * Searches for the maximum index in a vector.
     * 
     * @param value Vector to search.
     * @return The maximum index in the vector.
     */
    int SearchMaxIndex(const std::vector<double>& value);

    /**
     * @brief Prints the neuron values.
     * 
     * Outputs the neuron values to the console.
     * 
     * @param layers Number of layers to print.
     */
    void PrintValues(int layers);

    /**
     * @brief Performs backpropagation.
     * 
     * Performs backpropagation through the network.
     * 
     * @param expect Expected output value.
     */
    void BackPropogation(double expect);

    /**
     * @brief Updates the weights.
     * 
     * Updates the weights of the network using the specified learning rate.
     * 
     * @param lr Learning rate.
     */
    void WeightsUpdater(double lr);

    /**
     * @brief Saves the weights to a file.
     * 
     * Saves the weights of the network to a file.
     */
    void SaveWeights();

    /**
     * @brief Reads the weights from a file.
     * 
     * Reads the weights of the network from a file.
     */
    void ReadWeights();
};

