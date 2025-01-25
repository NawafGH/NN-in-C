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
 * init_network
 * ------------
 * Allocates arrays for layer_sizes, weights, and biases, and then
 * initializes them (random or zeros).
 */
void init_network(NeuralNet *net, int num_layers, const int *layer_sizes)
{
    net->num_layers = num_layers;

    // 1) Copy the layer_sizes array
    net->layer_sizes = (int*) malloc(num_layers * sizeof(int));
    for(int i = 0; i < num_layers; i++){
        net->layer_sizes[i] = layer_sizes[i];
    }

    // 2) Allocate weights and biases
    //    We have (num_layers - 1) weight matrices and (num_layers - 1) bias arrays
    net->weights = (float***) malloc((num_layers - 1) * sizeof(float**));
    net->biases  = (float**)  malloc((num_layers - 1) * sizeof(float*));

    // 3) For each connection from layer i to i+1:
    //    weights[i] is a 2D array of size [layer_sizes[i+1]] x [layer_sizes[i]]
    //    biases[i]  is a 1D array of size [layer_sizes[i+1]]
    for(int i = 0; i < num_layers - 1; i++){
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