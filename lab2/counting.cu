#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

void CountPosition1(const char *text, int *pos, int text_size)
{
    thrust::device_ptr<const char> dev_text = \
            thrust::device_pointer_cast(text);
    thrust::device_ptr<int> dev_pos = \
            thrust::device_pointer_cast(pos);
    char *space;
    cudaMalloc((void **) &space, sizeof(char)*text_size);
    cudaMemset(space, 10, text_size);
    thrust::device_ptr<const char> device_space_ptr = \
            thrust::device_pointer_cast(space); 
    thrust::transform(dev_text,
                    dev_text + text_size,
                    device_space_ptr,
                    dev_pos,
                    thrust::not_equal_to< const char >());
    thrust::inclusive_scan_by_key(dev_pos, 
                                dev_pos + text_size, 
                                dev_pos, 
                                dev_pos);
}

__global__
void kernel(const char *text, int *pos, int text_size) 
{
    int i;
    for (i = 0; i < text_size; i++) {
        if (text[i] != '\n') {
            if (i == 0) {
                pos[i] = 1;
            } else {
                pos[i] = pos[i-1] + 1;
            }
        } else {
            pos[i] = 0;
        }
    }
}

void CountPosition2(const char *text, int *pos, int text_size)
{
    kernel<<<1, 1>>>(text, pos, text_size);
}