#ifndef HOST_ONLY_H_
#define HOST_ONLY_H_

#include "init.h"
#include <uchar.h>

#include <sys/time.h>
#include <stdbool.h>
#include <getopt.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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

// Number of samples in each channel's dataset
extern int32_t number_of_input_samples;

extern hdc_vars hd;

extern uint32_t iM[MAX_IM_LENGTH * (MAX_BIT_DIM + 1)];

extern uint32_t chAM[MAX_CHANNELS * (MAX_BIT_DIM + 1)];

int
read_data(char const *input_file, double **test_set);
int
round_to_int(double num);
void
quantize_set(double const *input_set, int32_t *buffer);
void
nomem();
#endif
