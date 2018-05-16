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

__global__ void transform(const char *text, int *pos, int text_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < text_size) {
        pos[tid] = int (text[tid] != '\n');
    }
}

__global__
void kernel(const char *text, int *pos, int text_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        if (pos[tid] == 1) {
            int tmp = 1;
            while (pos[tid + tmp] != 0) {
                pos[tid + tmp] = pos[tid + tmp - 1] + 1;
                tmp++;
            }
        }
    } else {
        if (tid < text_size) {
            if (pos[tid] == 1) {
                if (pos[tid - 1] == 0) {
                    int tmp = 1;
                    while (pos[tid + tmp] != 0) {
                        pos[tid + tmp] = pos[tid + tmp - 1] + 1;
                        tmp++;
                    }
                }
            }
        }
    }
}

void CountPosition2(const char *text, int *pos, int text_size)
{
    int num_block = CeilDiv(text_size, 32);
    transform<<< num_block, 32 >>>(text, pos, text_size);
    kernel<<< num_block, 32 >>>(text, pos, text_size);

}