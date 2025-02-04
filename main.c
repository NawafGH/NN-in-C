#include <stdio.h>
#include <stdlib.h>
#include <time.h>        // for time() in srand()
#include "data.h"
#include "neuralnet.h"

int main() {
    // 1) Seed the random number generator for random weight initialization
    srand((unsigned int)time(NULL));

    // 2) Load MNIST training data
    Dataset train_data;
    if (load_mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte", &train_data) != 0) {
        printf("Failed to load MNIST training data.\n");
        return 1;
    }
    printf("Loaded %d training samples, each with %d features.\n",
           train_data.num_samples, train_data.num_features);

    // 3) Load MNIST test data
    Dataset test_data;
    if (load_mnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", &test_data) != 0) {
        printf("Failed to load MNIST test data.\n");
        free_dataset(&train_data);
        return 1;
    }
    printf("Loaded %d test samples, each with %d features.\n",
           test_data.num_samples, test_data.num_features);

    // 4) Create Neural Network
    NeuralNet net;
    int num_layers = 3;
    int layer_sizes[] = { 784, 128, 10 }; // MNIST: 784 input, 128 hidden, 10 output
    init_network(&net, num_layers, layer_sizes);

    // 5) Training parameters
    int epochs = 5;            // Run a few epochs
    float learning_rate = 0.01f;

    // 6) Training loop
    for (int e = 0; e < epochs; e++) {
        float total_loss = 0.0f;
        int correct = 0;

        for (int i = 0; i < train_data.num_samples; i++) {
            // Convert label to one-hot vector
            int label_int = (int)train_data.labels[i];
            float target[10] = {0};
            target[label_int] = 1.0f;

            // Perform backpropagation
            backprop(&net, train_data.features[i], target, learning_rate);

            // Compute loss & accuracy
            float output[10];
            forward(&net, train_data.features[i], output);

            float sample_loss = 0.0f;
            for (int j = 0; j < 10; j++) {
                float diff = output[j] - target[j];
                sample_loss += diff * diff;
            }
            total_loss += sample_loss;

            // Determine predicted label (argmax)
            int predicted = 0;
            float max_val = output[0];
            for (int j = 1; j < 10; j++) {
                if (output[j] > max_val) {
                    max_val = output[j];
                    predicted = j;
                }
            }
            if (predicted == label_int) {
                correct++;
            }
        }

        // Print training stats
        float avg_loss = total_loss / train_data.num_samples;
        float accuracy = 100.0f * (float)correct / (float)train_data.num_samples;
        printf("Epoch %d/%d - Avg Loss: %.4f - Accuracy: %.2f%%\n",
               (e + 1), epochs, avg_loss, accuracy);
    }

    // 7) Test Model on Unseen Data
    printf("\nEvaluating on Test Set...\n");

    int test_correct = 0;
    for (int i = 0; i < test_data.num_samples; i++) {
        float output[10];
        forward(&net, test_data.features[i], output);

        // Determine predicted label
        int predicted = 0;
        float max_val = output[0];
        for (int j = 1; j < 10; j++) {
            if (output[j] > max_val) {
                max_val = output[j];
                predicted = j;
            }
        }

        // Compare with ground truth
        int true_label = (int)test_data.labels[i];
        if (predicted == true_label) {
            test_correct++;
        }
    }

    // Print test accuracy
    float test_accuracy = 100.0f * (float)test_correct / (float)test_data.num_samples;
    printf("Test Accuracy: %.2f%% (%d/%d correct)\n",
           test_accuracy, test_correct, test_data.num_samples);

    // 8) Cleanup
    free_network(&net);
    free_dataset(&train_data);
    free_dataset(&test_data);

    return 0;
}
