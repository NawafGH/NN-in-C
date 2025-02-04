/* data.c */

#include "data.h"
#include <stdlib.h>  // for free
#include <stdio.h>

void free_dataset(Dataset* ds) {
    if (!ds) return;

    // Free the features 2D array if allocated
    if (ds->features) {
        for (int i = 0; i < ds->num_samples; i++) {
            free(ds->features[i]);
        }
        free(ds->features);
        ds->features = NULL;
    }

    // Free the labels array
    if (ds->labels) {
        free(ds->labels);
        ds->labels = NULL;
    }

    ds->num_samples = 0;
    ds->num_features = 0;
    ds->num_classes = 0;
}

// Helper to reverse endian for int
static int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

int load_mnist(const char *image_filepath, const char *label_filepath, Dataset *ds) {
    // Open image file
    FILE *img_fp = fopen(image_filepath, "rb");
    if (!img_fp) {
        fprintf(stderr, "Cannot open MNIST image file: %s\n", image_filepath);
        return 1;
    }

    // Open label file
    FILE *lbl_fp = fopen(label_filepath, "rb");
    if (!lbl_fp) {
        fprintf(stderr, "Cannot open MNIST label file: %s\n", label_filepath);
        fclose(img_fp);
        return 1;
    }

    // Read image file header
    int magic = 0, num_images = 0, rows = 0, cols = 0;
    fread(&magic, sizeof(magic), 1, img_fp);
    magic = reverse_int(magic);
    if (magic != 2051) {
        fprintf(stderr, "Invalid MNIST image magic: %d\n", magic);
        fclose(img_fp); fclose(lbl_fp);
        return 1;
    }

    fread(&num_images, sizeof(num_images), 1, img_fp);
    num_images = reverse_int(num_images);

    fread(&rows, sizeof(rows), 1, img_fp);
    rows = reverse_int(rows);

    fread(&cols, sizeof(cols), 1, img_fp);
    cols = reverse_int(cols);

    // Read label file header
    int lbl_magic = 0, num_labels = 0;
    fread(&lbl_magic, sizeof(lbl_magic), 1, lbl_fp);
    lbl_magic = reverse_int(lbl_magic);
    if (lbl_magic != 2049) {
        fprintf(stderr, "Invalid MNIST label magic: %d\n", lbl_magic);
        fclose(img_fp); fclose(lbl_fp);
        return 1;
    }

    fread(&num_labels, sizeof(num_labels), 1, lbl_fp);
    num_labels = reverse_int(num_labels);

    if (num_images != num_labels) {
        fprintf(stderr, "Warning: #images (%d) != #labels (%d)\n", num_images, num_labels);
        // We'll proceed, but in real code you might handle or unify
    }

    // Fill out the Dataset struct
    ds->num_samples  = num_images;
    ds->num_features = rows * cols;    // e.g. 28*28 = 784
    ds->num_classes  = 10;            // MNIST digits 0..9

    // Allocate ds->features
    ds->features = (float**) malloc(num_images * sizeof(float*));
    if (!ds->features) {
        fprintf(stderr, "Failed to allocate ds->features\n");
        fclose(img_fp); fclose(lbl_fp);
        return 1;
    }
    for (int i = 0; i < num_images; i++) {
        ds->features[i] = (float*) malloc(ds->num_features * sizeof(float));
        if (!ds->features[i]) {
            fprintf(stderr, "Failed to allocate ds->features[%d]\n", i);
            fclose(img_fp); fclose(lbl_fp);
            return 1;
        }
    }

    // Allocate ds->labels (1D array of floats, though these are really ints 0..9)
    ds->labels = (float*) malloc(num_images * sizeof(float));
    if (!ds->labels) {
        fprintf(stderr, "Failed to allocate ds->labels\n");
        fclose(img_fp); fclose(lbl_fp);
        return 1;
    }

    // Read image data
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < ds->num_features; j++) {
            unsigned char pixel = 0;
            fread(&pixel, 1, 1, img_fp);
            ds->features[i][j] = (float)pixel / 255.0f; // normalized [0..1]
        }
    }

    // Read label data
    for (int i = 0; i < num_labels; i++) {
        unsigned char lbl = 0;
        fread(&lbl, 1, 1, lbl_fp);
        ds->labels[i] = (float)lbl; // store as float, e.g., 3.0 for digit '3'
    }

    fclose(img_fp);
    fclose(lbl_fp);
    return 0; // success
}

