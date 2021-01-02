#include "init.h"
#include <uchar.h>

#include <sys/time.h>
#include <stdbool.h>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <math.h>
#include <helper_cuda.h>

#define TIME_DIFFERENCE(_start, _end) \
    ((_end.tv_sec + _end.tv_nsec / 1.0e9) - \
    (_start.tv_sec + _start.tv_nsec / 1.0e9))

#define TIME_DIFFERENCE_NSEC(_start, _end) \
    ((_end.tv_nsec < _start.tv_nsec)) ? \
    ((_end.tv_sec - 1 - (_start.tv_sec)) * 1e9 + _end.tv_nsec + 1e9 - _start.tv_nsec) : \
    ((_end.tv_sec - (_start.tv_sec)) * 1e9 + _end.tv_nsec - _start.tv_nsec)

#define TIME_DIFFERENCE_GETTIMEOFDAY(_start, _end) \
    ((_end.tv_sec + _end.tv_usec / 1.0e6) - \
    (_start.tv_sec + _start.tv_usec / 1.0e6))


#define TIME_NOW(_t) (clock_gettime(CLOCK_MONOTONIC, (_t)))

#define BUILTIN_CAO

__managed__ hdc_vars hd;

// Number of samples stored in each channel in the dataset
// In the test data sets, these are EMG samples which have been
// sampled at a rate of 500 Hertz
__managed__ int32_t number_of_input_samples;

//__managed__ uint32_t iM[MAX_IM_LENGTH * (MAX_BIT_DIM + 1)];
__managed__ uint32_t *iM;

//__managed__ uint32_t chAM[MAX_CHANNELS * (MAX_BIT_DIM + 1)];
__managed__ __device__ uint32_t *chAM;

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
    //*test_set = (double *)malloc(sz);
    checkCudaErrors(cudaMallocManaged(&(*test_set), sz));
    if (*test_set == NULL) {
        nomem();
    }
    if (fread(*test_set, 1, sz, file) != sz) {
        return ferror(file);
    }

    sz = hd.channels * (hd.bit_dim + 1) * sizeof(uint32_t);
    size_t bread;
    bread = fread(chAM, 1, sz, file);

    if (bread != sz) {
        return ferror(file);
    }

    sz = hd.im_length * (hd.bit_dim + 1) * sizeof(uint32_t);
    bread = fread(iM, 1, sz, file);

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

/**
 * @brief Tests the accuracy based on input testing queries.
 *
 * @param[in] q_32  Query hypervector
 * @param[in] aM_32 Trained associative memory
 * @return          Classification result
 */
__host__ __device__ int
associative_memory_32bit(uint32_t *q_32, uint32_t *aM_32) {
    int sims[CLASSES] = {0};

    // Computes Hamming Distances
    hamming_dist(q_32, aM_32, sims);

    // Classification with Hamming Metric
    return max_dist_hamm(sims);
}

/**
 * @brief Computes the maximum Hamming Distance.
 *
 * @param[in] distances Distances associated to each class
 * @return              The class related to the maximum distance
 */
__host__ __device__ int
max_dist_hamm(int *distances) {
    int max = distances[0];
    int max_index = 0;

    for (int i = 1; i < CLASSES; i++) {
        if (max > distances[i]) {
            max = distances[i];
            max_index = i;
        }
    }

    return max_index;
}

/**
 * @brief Computes the Hamming Distance for each class.
 *
 * @param[in] q     Query hypervector
 * @param[in] aM    Associative Memory matrix
 * @param[out] sims Distances' vector
 */
__host__ __device__ void
hamming_dist(uint32_t *q, uint32_t *aM, int *sims) {
    for (int i = 0; i < CLASSES; i++) {
        sims[i] = 0;
        for (int j = 0; j < hd.bit_dim + 1; j++) {
            sims[i] += number_of_set_bits(q[j] ^ aM[A2D1D(hd.bit_dim + 1, i, j)]);
        }
    }
}

/**
 * @brief Read from im
 * @param[in] im_ind    im array index
 */
__host__ __device__ static inline uint32_t
read_im(uint32_t im_ind) {
    return iM[im_ind];
}

/**
 * @brief Read from cham
 * @param[in] cham_ind    cham array index
 */
__host__ __device__ static inline uint32_t
read_cham(uint32_t cham_ind) {
    return chAM[cham_ind];
}

/**
 * @brief Computes the N-gram.
 *
 * @param[in] input       Input data
 * @param[out] query      Query hypervector
 */
__host__ __device__ void
compute_N_gram(int32_t *input, uint32_t *query) {

    uint32_t chHV[MAX_CHANNELS + 1];

    for (int i = 0; i < hd.bit_dim + 1; i++) {
        query[i] = 0;
        for (int j = 0; j < hd.channels; j++) {
            int ix = input[j];

            uint32_t im = read_im(A2D1D(hd.bit_dim + 1, ix, i));
            uint32_t cham = read_cham(A2D1D(hd.bit_dim + 1, j, i));

            chHV[j] = im ^ cham;
        }
        // this is done to make the dimension of the matrix for the componentwise majority odd.
        chHV[hd.channels] = chHV[0] ^ chHV[1];

        // componentwise majority: compute the number of 1's
        for (int z = 31; z >= 0; z--) {
            uint32_t cnt = 0;
            for (int j = 0; j < hd.channels + 1; j++) {
                uint32_t a = chHV[j] >> z;
                uint32_t mask = a & 1;
                cnt += mask;
            }

            if (cnt > 2) {
                query[i] = query[i] | (1 << z);
            }
        }
    }
}

/**
 * @brief Computes the number of 1's
 *
 * @param i The i-th variable that composes the hypervector
 * @return  Number of 1's in i-th variable of hypervector
 */
__host__ __device__ inline int
number_of_set_bits(uint32_t i) {
#ifdef BUILTIN_CAO
#ifdef  __CUDA_ARCH__
    return __popc(i);
#else
    return __builtin_popcount(i);
#endif
#else
    i = i - ((i >> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
#endif
}


/**
 * @struct hdc_data
 * @brief HDC data for HDC task
 */
typedef struct hdc_data {
    int32_t *data_set;     /**< Input HDC dataset */
    int32_t *results;      /**< Output from run */
    uint32_t result_len;   /**< Length of the results */
    double execution_time; /**< Total execution time of run */
} hdc_data;

/**
 * @brief Function for @p run_hdc to run HDC task
 */
typedef int (*hdc)(int32_t *data_set, int32_t *results, void *runtime);

/**
 * @brief Run the HDC algorithm for the host
 *
 * @param[in]  data_set  Input dataset
 * @param[out] results   Results from run
 * @param[out] runtime   Runtimes of individual sections (unused)
 *
 * @return               Non-zero on failure.
 */
static int
host_hdc(int32_t *data_set, int32_t *results, void *runtime) {

    (void) runtime;

    uint32_t overflow = 0;
    uint32_t old_overflow = 0;
    uint32_t mask = 1;
    uint32_t q[hd.bit_dim + 1];
    uint32_t q_N[hd.bit_dim + 1];
    int32_t quantized_buffer[hd.channels];

    int result_num = 0;

    for (int ix = 0; ix < number_of_input_samples; ix += hd.n) {

        for (int z = 0; z < hd.n; z++) {

            for (int j = 0; j < hd.channels; j++) {
                if (ix + z < number_of_input_samples) {
                    int ind = A2D1D(number_of_input_samples, j, ix + z);
                    quantized_buffer[j] = data_set[ind];
                }
            }

            // Spatial and Temporal Encoder: computes the n-gram.
            // N.B. if n = 1 we don't have the Temporal Encoder but only the Spatial Encoder.
            if (z == 0) {
                compute_N_gram(quantized_buffer, q);
            } else {
                compute_N_gram(quantized_buffer, q_N);

                // Here the hypervector q is shifted by 1 position as permutation,
                // before performing the componentwise XOR operation with the new query (q_N).
                overflow = q[0] & mask;

                for (int i = 1; i < hd.bit_dim; i++) {
                    old_overflow = overflow;
                    overflow = q[i] & mask;
                    q[i] = (q[i] >> 1) | (old_overflow << (32 - 1));
                    q[i] = q_N[i] ^ q[i];
                }

                old_overflow = overflow;
                overflow = (q[hd.bit_dim] >> 16) & mask;
                q[hd.bit_dim] = (q[hd.bit_dim] >> 1) | (old_overflow << (32 - 1));
                q[hd.bit_dim] = q_N[hd.bit_dim] ^ q[hd.bit_dim];

                q[0] = (q[0] >> 1) | (overflow << (32 - 1));
                q[0] = q_N[0] ^ q[0];
            }
        }
        // classifies the new N-gram through the Associative Memory matrix.
        results[result_num++] = associative_memory_32bit(q, hd.aM_32);
    }

    return 0;
}


__global__ void hdc_kernel(int32_t *data_set, int32_t *results, void *runtime)
{
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx % hd.n == 0) {
        uint32_t overflow = 0;
        uint32_t old_overflow = 0;
        uint32_t mask = 1;
        uint32_t *q = (uint32_t *)malloc(sizeof(uint32_t)*hd.bit_dim + 1);
        uint32_t *q_N = (uint32_t *)malloc(sizeof(uint32_t)*hd.bit_dim + 1);
        int32_t *quantized_buffer = (int32_t *)malloc(sizeof(int32_t)* hd.channels);

        //printf("%d %d %d %d\n",number_of_input_samples, hd.n, hd.channels, hd.bit_dim);
        
        for (int z = 0; z < hd.n; z++) {

            for (int j = 0; j < hd.channels; j++) {
                if (idx + z < number_of_input_samples) {
                    int ind = A2D1D(number_of_input_samples, j, idx + z);
                    quantized_buffer[j] = data_set[ind];
                }
            }

            // Spatial and Temporal Encoder: computes the n-gram.
            // N.B. if n = 1 we don't have the Temporal Encoder but only the Spatial Encoder.
            if (z == 0) {
                compute_N_gram(quantized_buffer, q);
            } else {
                compute_N_gram(quantized_buffer, q_N);

                // Here the hypervector q is shifted by 1 position as permutation,
                // before performing the componentwise XOR operation with the new query (q_N).
                overflow = q[0] & mask;

                for (int i = 1; i < hd.bit_dim; i++) {
                    old_overflow = overflow;
                    overflow = q[i] & mask;
                    q[i] = (q[i] >> 1) | (old_overflow << (32 - 1));
                    q[i] = q_N[i] ^ q[i];
                }

                old_overflow = overflow;
                overflow = (q[hd.bit_dim] >> 16) & mask;
                q[hd.bit_dim] = (q[hd.bit_dim] >> 1) | (old_overflow << (32 - 1));
                q[hd.bit_dim] = q_N[hd.bit_dim] ^ q[hd.bit_dim];

                q[0] = (q[0] >> 1) | (overflow << (32 - 1));
                q[0] = q_N[0] ^ q[0];
            }
        }
        // classifies the new N-gram through the Associative Memory matrix.
        results[idx/hd.n] = associative_memory_32bit(q, hd.aM_32);
    
        free(q);
        free(q_N);
        free(quantized_buffer);
    }
}

static int
gpu_hdc(int32_t *data_set, int32_t *results, void *runtime) {

    //int device = -1;
    //cudaGetDevice(&device);
    //cudaMemPrefetchAsync(iM, sizeof(uint32_t) * MAX_IM_LENGTH * (MAX_BIT_DIM + 1), device, NULL);
    //cudaMemPrefetchAsync(chAM, sizeof(uint32_t) * MAX_CHANNELS * (MAX_BIT_DIM + 1), device, NULL);

    hdc_kernel<<<number_of_input_samples,1>>>(data_set, results, runtime);
    checkCudaErrors(cudaDeviceSynchronize());

    return 0;
}

/**
 * @brief Run a HDC workload and time the execution
 *
 * @param[in] fn        Function to run HDC algorithm
 * @param[out] data     Results from HDC run
 * @param[out] runtime  Run times from sections of @p fn
 *
 * @return Non-zero On failure
 */
static double
run_hdc(hdc fn, hdc_data *data, void *runtime) {
    struct timespec start, end;

    int ret = 0;

    uint8_t extra_result = (number_of_input_samples % hd.n) != 0;
    data->result_len = (number_of_input_samples / hd.n) + extra_result;
    uint32_t result_size = data->result_len * sizeof(int32_t);


    //data->results = (int32_t *)malloc(result_size);
    checkCudaErrors(cudaMallocManaged(&data->results, result_size));
    
    if (data->results == NULL) {
        nomem();
    }

    TIME_NOW(&start);
    ret = fn(data->data_set, data->results, runtime);
    TIME_NOW(&end);

    data->execution_time = TIME_DIFFERENCE(start, end);

    return ret;
}

/**
 * @brief Compare the results from the host and GPU confirming they are the same
 *        or printing differences
 *
 * @param[in] gpu_data   Results to be tested from GPU
 * @param[in] host_data  Results to be tested from host
 * @param[in] check_only Only check results are equal, dont print differences
 *
 * @return               Non-zero if results are not the same
 */
static int
compare_results(hdc_data *gpu_data, hdc_data *host_data, bool check_only) {
    int ret = 0;

    if (!check_only) {
        printf("--- Compare --\n");
        printf("(%u) results\n", host_data->result_len);
    }

    for (uint32_t i = 0; i < host_data->result_len; i++) {
        if (host_data->results[i] != gpu_data->results[i]) {
            if (check_only) {
                return -1;
            }
            fprintf(stderr, "(host_results[%u] = %d) != (gpu_results[%u] = %d)\n", i,
                    host_data->results[i], i, gpu_data->results[i]);
            ret = -1;
        }
    }

    if (check_only) {
        return 0;
    }

  /*  char *faster;
    double time_diff, percent_diff;
    if (dpu_data->execution_time > host_data->execution_time) {
        faster = "Host";
        time_diff = dpu_data->execution_time - host_data->execution_time;
        percent_diff = dpu_data->execution_time / host_data->execution_time;
    } else {
        faster = "DPU";
        time_diff = host_data->execution_time - dpu_data->execution_time;
        percent_diff = host_data->execution_time / dpu_data->execution_time;
    }

    printf("%s was %fs (%f x) faster\n", faster, time_diff, percent_diff);
*/
    return ret;
}

/**
 * @brief Print results from HDC run
 * @param[in] data  Results to print
 */
static void
print_results(hdc_data *data) {
    for (uint32_t i = 0; i < data->result_len; i++) {
        printf("%d\n", data->results[i]);
    }
}

/**
 * @brief Display usage information to @p stream
 * @param[in] stream    File pointer to write usage to
 * @param[in] exe_name  Name of executable
 */
static void
usage(FILE *stream, char const *exe_name) {
#ifdef DEBUG
    fprintf(stream, "**DEBUG BUILD**\n");
#endif

    fprintf(stream, "usage: %s [ -d ] -i <INPUT_FILE>\n", exe_name);
    fprintf(stream, "\td: use GPU\n");
    fprintf(stream, "\ti: input file\n");
    fprintf(stream, "\tr: show runtime only\n");
    fprintf(stream, "\ts: show results\n");
    fprintf(stream, "\tt: test results\n");
    fprintf(stream, "\th: help message\n");
}

int
main(int argc, char **argv) {
    bool use_gpu = false;
    bool show_results = false;
    bool test_results = false;
    bool runtime_only = false;
    int ret = 0;
    int gpu_ret = 0;
    int host_ret = 0;
    char const options[] = "dsthri:";
    char *input = NULL;

    checkCudaErrors(cudaMallocManaged(&iM, sizeof(uint32_t) * MAX_IM_LENGTH * (MAX_BIT_DIM + 1)));
    checkCudaErrors(cudaMallocManaged(&chAM, sizeof(uint32_t) * MAX_CHANNELS * (MAX_BIT_DIM + 1)));

    int opt;
    while ((opt = getopt(argc, argv, options)) != -1) {
        switch (opt) {
            case 'd':
                use_gpu = true;
                break;

            case 'i':
                input = optarg;
                break;

            case 's':
                show_results = true;
                break;

            case 't':
                test_results = true;
                break;

            case 'r':
                runtime_only = true;
                break;

            case 'h':
                usage(stdout, argv[0]);
                return EXIT_SUCCESS;

            default:
                usage(stderr, argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (input == NULL) {
        fprintf(stderr, "Please add an input file\n");
        usage(stderr, argv[0]);
        return EXIT_FAILURE;
    }

    double *test_set;
    ret = read_data(input, &test_set);
    if (ret != 0) {
        return ret;
    }

    uint32_t buffer_size = (sizeof(int32_t) * number_of_input_samples * hd.channels);
    //int32_t *data_set = (int32_t *)malloc(buffer_size);
    int32_t *data_set;
    checkCudaErrors(cudaMallocManaged(&data_set, buffer_size));

    if (data_set == NULL) {
        nomem();
    }

    quantize_set(test_set, data_set);

    hdc_data gpu_results = {.data_set = data_set, .results = NULL};
    hdc_data host_results = {.data_set = data_set, .results = NULL};

    if (use_gpu || test_results) {
        gpu_ret = run_hdc(gpu_hdc, &gpu_results, NULL);
        if (gpu_ret != 0) {
            goto err;
        }
    }

    if (!use_gpu || test_results) {
        host_ret = run_hdc(host_hdc, &host_results, NULL);
        if (host_ret != 0) {
            goto err;
        }
    }

    if (!use_gpu || test_results) {
        if (!runtime_only) {
            printf("--- Host --\n");
            if (show_results) {
                print_results(&host_results);
            }
            printf("Host took %f\n", host_results.execution_time);
        } else {
            printf("%f\n", host_results.execution_time);
        }
    }

    if (use_gpu || test_results) {
        if (!runtime_only) {
            printf("--- GPU --\n");
            if (show_results) {
                print_results(&gpu_results);
            }
            printf("GPU took %f\n", gpu_results.execution_time);
        } else {
            printf("%f\n", gpu_results.execution_time);
        }
    }

    if (test_results) {
        ret = compare_results(&gpu_results, &host_results, runtime_only);
    }

err:
    //free(data_set);
    checkCudaErrors(cudaFree(data_set));
    //free(test_set);
    checkCudaErrors(cudaFree(test_set));
    //free(host_results.results);
    checkCudaErrors(cudaFree(host_results.results));
    //free(gpu_results.results);
    checkCudaErrors(cudaFree(gpu_results.results));

    checkCudaErrors(cudaFree(iM));
    checkCudaErrors(cudaFree(chAM));

    return (ret + gpu_ret + host_ret);
}
