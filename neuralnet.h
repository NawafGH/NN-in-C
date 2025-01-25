#ifndef NEURALNET_H
#define NEURALNET_H

typedef struct {
    int num_layers;       // total number of layers
    int *layer_sizes;     // array of layer sizes: length = num_layers

    // weights[i]: 2D array for the connection from layer i -> i+1
    //   dimension = [layer_sizes[i+1]][layer_sizes[i]]
    float ***weights;

    // biases[i]: 1D array for layer (i+1)
    //   dimension = [layer_sizes[i+1]]
    float **biases;

} NeuralNet;

/**
 * @brief Initializes a neural network with the given layer sizes.
 *
 * @param net           Pointer to a NeuralNet struct to be initialized
 * @param num_layers    How many layers (e.g. 3 for input-hidden-output)
 * @param layer_sizes   Array of integers specifying neurons in each layer
 *                      (e.g. [784, 128, 10] for an MLP).
 *
 * This function allocates memory for the weights and biases
 * (weights[i] for connecting layer i to i+1, biases[i] for layer i+1).
 * It also randomly initializes the weights (and typically zeros the biases).
 */
void init_network(NeuralNet *net, int num_layers, const int *layer_sizes);

/**
 * @brief Frees all dynamically allocated memory in the NeuralNet.
 *
 * @param net Pointer to a NeuralNet struct whose memory should be freed.
 *
 * After calling this, the NeuralNet struct won't be valid until reinitialized.
 */
void free_network(NeuralNet *net);

/**
 * @brief Performs a forward pass through the network.
 *
 * @param net      Pointer to the NeuralNet.
 * @param input    Pointer to an array of floats (size of layer_sizes[0]) representing one input sample.
 * @param output   Pointer to an array of floats (size of layer_sizes[num_layers-1]) 
 *                 where the final layer’s output will be stored.
 *
 * This function should compute the activations from the input layer
 * all the way to the output layer and store the final in `output`.
 */
void forward(const NeuralNet *net, const float *input, float *output);

/**
 * @brief Performs backpropagation and updates network weights/biases.
 *
 * @param net     Pointer to the NeuralNet (will be modified).
 * @param input   Pointer to an array of floats for the input (size = layer_sizes[0]).
 * @param target  Pointer to an array of floats for the target/label
 *                (size = layer_sizes[num_layers-1]).
 * @param lr      Learning rate (e.g. 0.01).
 *
 * This function typically:
 *  1. Computes forward pass
 *  2. Computes gradients (error signals at each layer)
 *  3. Updates `weights` and `biases` accordingly
 */
void backprop(NeuralNet *net, const float *input, const float *target, float lr);



#endif // NEURALNET_H