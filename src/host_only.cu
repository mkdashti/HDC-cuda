#include "host_only.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

dpu_hdc_vars hd;

// Number of samples stored in each channel in the dataset
// In the test data sets, these are EMG samples which have been
// sampled at a rate of 500 Hertz
int32_t number_of_input_samples;

#ifndef IM_IN_WRAM
uint32_t iM[MAX_IM_LENGTH * (MAX_BIT_DIM + 1)];
#endif

#ifndef CHAM_IN_WRAM
uint32_t chAM[MAX_CHANNELS * (MAX_BIT_DIM + 1)];
#endif

/**
 * @brief Exit if NOMEM
 */
void
nomem() {
    fprintf(stderr, "ERROR: No memory\n");
    exit(ENOMEM);
}

/**
 * @brief Read data from @p input_file into globals and test_set
 *
 * @param[in] input_file    File to read from
 * @param[in,out] test_set  Test data to allocate and fill
 */
int
read_data(char const *input_file, double **test_set) {
    int ret = 0;
    errno = 0;
    FILE *file = fopen(input_file, "rb");
    if (file == NULL) {
        return errno != 0 ? errno : -1;
    }

    int32_t version;
    size_t sz = sizeof(version);
    if (fread(&version, 1, sz, file) != sz) {
        return ferror(file);
    }
    if (version != VERSION) {
        fprintf(stderr, "Binary file version (%d) does not match expected (%d)\n", version,
                VERSION);
        return -1;
    }
    if ((fread(&hd.dimension, 1, sz, file) != sz) || (fread(&hd.channels, 1, sz, file) != sz) ||
        (fread(&hd.bit_dim, 1, sz, file) != sz) ||
        (fread(&number_of_input_samples, 1, sz, file) != sz) || (fread(&hd.n, 1, sz, file) != sz) ||
        (fread(&hd.im_length, 1, sz, file) != sz)) {
        return ferror(file);
    }

    sz = hd.channels * number_of_input_samples * sizeof(double);
    *test_set = (double *)malloc(sz);
    if (*test_set == NULL) {
        nomem();
    }
    if (fread(*test_set, 1, sz, file) != sz) {
        return ferror(file);
    }

    sz = hd.channels * (hd.bit_dim + 1) * sizeof(uint32_t);
    size_t bread;
#ifdef CHAM_IN_WRAM
    bread = fread(hd.chAM, 1, sz, file);
#else
    bread = fread(chAM, 1, sz, file);
#endif
    if (bread != sz) {
        return ferror(file);
    }

    sz = hd.im_length * (hd.bit_dim + 1) * sizeof(uint32_t);
#ifdef IM_IN_WRAM
    bread = fread(hd.iM, 1, sz, file);
#else
    bread = fread(iM, 1, sz, file);
#endif
    if (bread != sz) {
        return ferror(file);
    }

    sz = hd.n * (hd.bit_dim + 1) * sizeof(uint32_t);
    if (fread(hd.aM_32, 1, sz, file) != sz) {
        return ferror(file);
    }

    fclose(file);
    return ret;
}

/**
 * @brief Round a double to an integer.
 *
 * @param[in] num Double to round
 * @return        Rounded integer value
 */
int
round_to_int(double num) {
    return (num - floor(num) > 0.5) ? ceil(num) : floor(num);
}

/**
 * @brief Quantization: each sample is rounded to the nearest integer.
 *
 * @param[out] buffer Rounded integers
 */
void
quantize_set(double const *input_set, int32_t *buffer) {
    for (int i = 0; i < hd.channels; i++) {
        for (int j = 0; j < number_of_input_samples; j++) {
            buffer[(i * number_of_input_samples) + j] =
                round_to_int(input_set[(i * number_of_input_samples) + j]);
        }
    }
}
