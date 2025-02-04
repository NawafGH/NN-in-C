#include <stdio.h>      
#include <stdlib.h>     
#include <string.h>     
#include <math.h>       
#include "neuralnet.h"

//allocates memory for a 2d array
static float** allocate_2d_array(int rows, int cols) {
    float **arr = (float**) malloc(rows * sizeof(float*));
    if (!arr) {
        fprintf(stderr, "Error: failed to allocate memory for 2D array.\n");
        exit(EXIT_FAILURE);
    }
    for(int r = 0; r < rows; r++) {
        arr[r] = (float*) malloc(cols * sizeof(float));
        if (!arr[r]) {
            fprintf(stderr, "Error: failed to allocate memory for 2D sub-array.\n");
            exit(EXIT_FAILURE);
        }
    }
    return arr;
}

/*
 * A simple sigmoid activation function and its derivative.
 * You can replace these with ReLU, tanh, or anything else.
 */
static float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float sigmoidf_deriv(float x) {
    // derivative of sigmoid wrt x = sigmoid(x)*(1 - sigmoid(x))
    float s = sigmoidf(x);
    return s * (1.0f - s);
}

/*
 * init_network
 * ------------
 * Allocates arrays for layer_sizes, weights, and biases, and then
 * initializes them (random or zeros).
 */
void init_network(NeuralNet *net, int num_layers, const int *layer_sizes) {
    net->num_layers = num_layers;

    // 1) Copy the layer_sizes array
    net->layer_sizes = (int*) malloc(num_layers * sizeof(int));
    for(int i = 0; i < num_layers; i++) {
        net->layer_sizes[i] = layer_sizes[i];
    }

    // 2) Allocate weights and biases
    //    We have (num_layers - 1) weight matrices and (num_layers - 1) bias arrays
    net->weights = (float***) malloc((num_layers - 1) * sizeof(float**));
    net->biases  = (float**)  malloc((num_layers - 1) * sizeof(float*));

    // 3) For each connection from layer i to i+1:
    //    weights[i] is a 2D array of size [layer_sizes[i+1]] x [layer_sizes[i]]
    //    biases[i]  is a 1D array of size [layer_sizes[i+1]]
    for(int i = 0; i < num_layers - 1; i++) {
        int in_size = layer_sizes[i];
        int out_size = layer_sizes[i+1];
    
        // Allocate weights[i]
        net->weights[i] = allocate_2d_array(out_size, in_size);

        // Allocate biases[i]
        net->biases[i] = (float*) malloc(out_size * sizeof(float));

        // 4) Initialize weights randomly, biases to zero (or small random).
        // TODO: consider better init like Xavier or He.
        for(int out_n = 0; out_n < out_size; out_n++) {
            for(int in_n = 0; in_n < in_size; in_n++) {
                // random float in range [-0.5, 0.5]
                net->weights[i][out_n][in_n] = ((float) rand() / RAND_MAX) - 0.5f;
            }
            net->biases[i][out_n] = 0.0f; // or small random
        }
    }
}

/*
 * free_network
 * ------------
 * Frees all allocated memory in the NeuralNet.
 */
void free_network(NeuralNet *net) {
    // Free weights
    for (int i = 0; i < net->num_layers - 1; i++) {
        int out_size = net->layer_sizes[i+1];

        //free each row
        for(int row = 0; row < out_size; row++) {
            free(net->weights[i][row]);
        }
        free(net->weights[i]);
    }
    free(net->weights);
    

    // Free biases
    for(int i = 0; i < net->num_layers - 1; i++) {
        free(net->biases[i]);
    }
    free(net->biases);

    // Free layer_sizes
    free(net->layer_sizes);

    // Set pointers to NULL to avoid dangling references
    net->weights = NULL;
    net->biases  = NULL;
    net->layer_sizes = NULL;
    net->num_layers  = 0;
}

/*
 * forward
 * -------
 * Perform a forward pass through the network with a sigmoid activation.
 * 
 * input: array of floats, size = layer_sizes[0]
 * output: array of floats, size = layer_sizes[num_layers - 1]
 */
void forward(const NeuralNet *net, const float *input, float *output) {
    // We'll allocate temporary arrays to store the activation at each layer.
    // Alternatively, you could store all layer outputs in a 2D array. But let's keep it simple.
    float *curr_in  = (float*) malloc(net->layer_sizes[0] * sizeof(float));
    float *curr_out = NULL;  // for each subsequent layer

    // Copy the input into curr_in
    for(int i = 0; i < net->layer_sizes[0]; i++) {
        curr_in[i] = input[i];
    }

    // Forward through each set of weights
    for(int layer_idx = 0; layer_idx < net->num_layers - 1; layer_idx++) {
        int in_size  = net->layer_sizes[layer_idx];
        int out_size = net->layer_sizes[layer_idx + 1];

        // Allocate space for layer output
        curr_out = (float*) malloc(out_size * sizeof(float));

        // For each neuron in the next layer:
        for(int out_n = 0; out_n < out_size; out_n++) {
            float sum = 0.0f;

            // Weighted sum from all inputs
            for(int in_n = 0; in_n < in_size; in_n++) {
                sum += net->weights[layer_idx][out_n][in_n] * curr_in[in_n];
            }
            sum += net->biases[layer_idx][out_n];

            // Apply activation (sigmoid here)
            curr_out[out_n] = sigmoidf(sum);
        }

        // curr_out is now the input to the next layer
        free(curr_in); // free the previous layer's memory
        curr_in = curr_out; // rename for next iteration
    }

    // curr_in now holds the final layer's output
    // copy it to the user-provided output array
    int final_size = net->layer_sizes[net->num_layers - 1];
    for(int i = 0; i < final_size; i++) {
        output[i] = curr_in[i];
    }

    // free the last curr_in
    free(curr_in);
}

/*
 * backprop
 * --------
 * Performs a simple backprop using Mean Squared Error (MSE) 
 * for demonstration. 
 * 
 * For a single sample (input, target):
 *  1. Forward pass (store intermediate activations).
 *  2. Compute output error and propagate backward.
 *  3. Update weights and biases with gradient * lr.
 *
 * This is a minimal example, not covering every best practice. 
 */
void backprop(NeuralNet *net, const float *input, const float *target, float lr) {
    /* ---- Step 1: Forward pass storing all layer activations ---- */

    // We’ll store activation for each layer in an array of float*.
    // activations[i] will be array of size layer_sizes[i].
    float **activations = (float**) malloc(net->num_layers * sizeof(float*));
    if(!activations) {
        fprintf(stderr, "Error: failed to allocate memory for activations.\n");
        exit(EXIT_FAILURE);
    }

    // Allocate and copy input as activations[0]
    activations[0] = (float*) malloc(net->layer_sizes[0] * sizeof(float));
    for(int i = 0; i < net->layer_sizes[0]; i++) {
        activations[0][i] = input[i];
    }

    // Forward pass to fill activations[i] for i=1..(num_layers-1)
    for(int layer_idx = 0; layer_idx < net->num_layers - 1; layer_idx++) {
        int in_size  = net->layer_sizes[layer_idx];
        int out_size = net->layer_sizes[layer_idx + 1];

        activations[layer_idx + 1] = (float*) malloc(out_size * sizeof(float));

        for(int out_n = 0; out_n < out_size; out_n++) {
            float sum = 0.0f;
            // Weighted sum from previous layer
            for(int in_n = 0; in_n < in_size; in_n++) {
                sum += net->weights[layer_idx][out_n][in_n] * activations[layer_idx][in_n];
            }
            sum += net->biases[layer_idx][out_n];

            // activation function
            activations[layer_idx + 1][out_n] = sigmoidf(sum);
        }
    }

    /* ---- Step 2: Compute gradients (error) and backpropagate ---- */
    // nno
    // We'll keep an array of "delta" for each layer (except input).
    // delta[i] = derivative of loss wrt (z_i) for each neuron in layer i
    // z_i is the weighted sum before activation, but for simplicity, 
    // we re-derive it from the activation function derivative approach.
    float **delta = (float**) malloc((net->num_layers) * sizeof(float*));
    if(!delta) {
        fprintf(stderr, "Error: failed to allocate memory for delta.\n");
        exit(EXIT_FAILURE);
    }

    // Allocate for output layer
    int output_layer_idx = net->num_layers - 1;
    int output_size = net->layer_sizes[output_layer_idx];
    delta[output_layer_idx] = (float*) malloc(output_size * sizeof(float));

    // If using Mean Squared Error:
    // dL/d(a_out) = (a_out - target)
    // d(a_out)/d(z_out) = sigmoid'(z_out) = a_out * (1 - a_out)
    // so delta_out = (a_out - target) * (a_out*(1-a_out))
    for(int j = 0; j < output_size; j++) {
        float a_out = activations[output_layer_idx][j];
        float error = a_out - target[j]; 
        delta[output_layer_idx][j] = error * (a_out * (1.0f - a_out));
    }

    // Now backprop for hidden layers
    // delta[l] = (W[l]^T * delta[l+1]) . * sigmoid'(z_l)
    // but we'll reconstruct z_l from the activation: 
    //   z_l = a_l * (1 - a_l) if using sigmoid derivative
    for(int layer_idx = net->num_layers - 2; layer_idx > 0; layer_idx--) {
        int layer_size     = net->layer_sizes[layer_idx];
        int next_layer_size= net->layer_sizes[layer_idx + 1];

        delta[layer_idx] = (float*) malloc(layer_size * sizeof(float));

        for(int i = 0; i < layer_size; i++) {
            // Weighted sum of errors from next layer
            float sum_error = 0.0f;
            for(int k = 0; k < next_layer_size; k++) {
                sum_error += net->weights[layer_idx][k][i] * delta[layer_idx + 1][k];
            }
            float a = activations[layer_idx][i]; 
            float d_act = a * (1.0f - a); // derivative of sigmoid
            delta[layer_idx][i] = sum_error * d_act;
        }
    }

    /* ---- Step 3: Update weights and biases ---- */
    // W[l][out_n][in_n] -= lr * (delta[l+1][out_n] * a_l[in_n])
    // b[l][out_n]       -= lr * delta[l+1][out_n]
    for(int layer_idx = 0; layer_idx < net->num_layers - 1; layer_idx++) {
        int in_size  = net->layer_sizes[layer_idx];
        int out_size = net->layer_sizes[layer_idx + 1];

        for(int out_n = 0; out_n < out_size; out_n++) {
            float d = delta[layer_idx + 1][out_n]; // error for that neuron
            // update each weight
            for(int in_n = 0; in_n < in_size; in_n++) {
                float a_in = activations[layer_idx][in_n]; 
                net->weights[layer_idx][out_n][in_n] -= lr * (d * a_in);
            }
            // update bias
            net->biases[layer_idx][out_n] -= lr * d;
        }
    }

    /* ---- Cleanup: free memory used by backprop ---- */
    for(int i = 0; i < net->num_layers; i++) {
        free(activations[i]);
    }
    free(activations);

    for(int i = 1; i < net->num_layers; i++) {
        free(delta[i]);
    }
    free(delta);
}