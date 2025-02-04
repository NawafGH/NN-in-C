/* data.h */

#ifndef DATA_H
#define DATA_H

/* 
 * For classification tasks, you might store labels as integers (e.g., 0..9).
 * For regression tasks, you might store labels as floats.
 * For multi-label or multi-output, you might store "labels" as a 2D float array.
 *
 * Here's a common structure that can handle many scenarios.
 */
typedef struct {
    float **features;   // 2D array [num_samples][num_features]
    float *labels;      // 1D array [num_samples], 
                        //   or you might use float** if you have multi-dimensional labels

    int num_samples;    // How many samples (rows) do we have?
    int num_features;   // How many features (columns) for each sample?
    int num_classes;    // (Optional) for classification tasks
                        // e.g., 10 for MNIST digits. If not relevant, set 0 or 1.

    // Additional metadata fields are possible:
    // e.g., image_width, image_height (for image data),
    // or column names if CSV, etc.
} Dataset;

/**
 * @brief Frees all memory allocated inside a Dataset struct.
 */
void free_dataset(Dataset* ds);

/* 
 * You might have specialized load functions or a single load_data function
 * that uses an enum or flags to decide how to parse.
 */

/**
 * @brief Loads MNIST from the given image and label files into a generic Dataset.
 *
 * @param image_filepath Path to the MNIST image file (e.g., train-images.idx3-ubyte).
 * @param label_filepath Path to the MNIST label file (e.g., train-labels.idx1-ubyte).
 * @param ds            Pointer to a Dataset struct to fill.
 *
 * @return 0 on success, non-zero on error.
 */
int load_mnist(const char *image_filepath, const char *label_filepath, Dataset *ds);

/**
 * @brief Loads data from a CSV file into a generic Dataset (example).
 *
 * @param filepath     Path to the CSV file.
 * @param ds           Pointer to a Dataset struct to fill.
 * @param has_header   If 1, skip the first line. If 0, treat first line as data.
 * @param label_column Index of the column to treat as the label (if any).
 *
 * @return 0 on success, non-zero on error.
 */
int load_csv(const char *filepath, Dataset *ds, int has_header, int label_column);

/* etc. ... */

#endif // DATA_H
