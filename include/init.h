#ifndef INIT_H_
#define INIT_H_

#include <stdint.h>

// 2d array to 1d array index
#define A2D1D(d1, i0, i1) (((d1) * (i0)) + (i1))

// Expected versioned binary format (first 4 bytes)
#define VERSION 0

// CHANNELS_VOTING for the componentwise majority must be odd
#define CHANNELS_VOTING (channels + 1)

// Number of CLASSES to be classified
#define CLASSES 5

// Sample size max per DPU in each channel in 32 bit integers (make sure aligned bytes)
#define SAMPLE_SIZE_MAX 512

/**
 * @struct hdc_vars
 * @brief HDC specific data and variables
 */
typedef struct hdc_vars {
    int32_t dimension; /**< Dimension of the hypervectors */
    int32_t channels;  /**< Number of acquisition's CHANNELS */
    int32_t bit_dim;   /**< Dimension of the hypervectors after compression */
    int32_t n;         /**< Dimension of the N-grams */
    int32_t im_length; /**< Item memory length */
    uint32_t aM_32[MAX_N * (MAX_BIT_DIM + 1)]; /**< Associative memory */
} hdc_vars;

#endif // INIT_H_
