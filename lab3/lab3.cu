#include "lab3.h"
#include <cstdio>

#define CUR curt*3
#define LEFT curt*3-3
#define RIGHT curt*3+3
#define UP curt*3-3*wt
#define DOWN curt*3+3*wt

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__global__ void CalculateFixed (
	const float *background,
	const float *target,
	const float *mask,
	float *fixed,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	float tmp[3] = {0.0, 0.0, 0.0};
	const int yb = oy+yt, xb = ox+xt;
	const int curb = wb*yb+xb;
	if ( (xt < (wt-1)) and  (yt < (ht-1)) and (xt > 0) and (yt > 0) ) {
		// general case
		// determine four neighbors' mask
		if (mask[curt-1] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 - 3 + c];
			}
		}
		if (mask[curt+1] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 + 3 + c];
			}
		}
		if (mask[curt-wt] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 - 3*wb + c];
			}
		}
		if (mask[curt+wt] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 + 3*wb + c];
			}
		}
		for (int c = 0; c < 3; ++c) {
			fixed[CUR+c] = 4 * target[CUR+c] - \
				((target[LEFT+c]) + (target[RIGHT+c]) + \
				(target[UP+c]) + (target[DOWN+c])) + tmp[c];
		}
	} else if ( (xt == 0) and (yt == 0) ) {
		// left-up corner
		if (mask[curt+1] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 + 3 + c];
			}
		}
		if (mask[curt+wt] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 + 3*wb + c];
			}
		}
		for (int c = 0; c < 3; ++c) {
			tmp[c] += background[curb*3 - 3*wb + c];
			tmp[c] += background[curb*3 - 3 + c];
		}
		for (int c = 0; c < 3; ++c) {
			fixed[CUR+c] = 2 * target[CUR+c] - \
				((target[RIGHT+c]) + \
				(target[DOWN+c])) + tmp[c];
		}
	} else if ( (xt == (wt-1)) and (yt == 0) ) {
		// right-up corner
		if (mask[curt+wt] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 + 3*wb + c];
			}
		}
		if (mask[curt-1] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 - 3 + c];
			}
		}
		for (int c = 0; c < 3; ++c) {
			tmp[c] += background[curb*3 - 3*wb + c];
			tmp[c] += background[curb*3 + 3 + c];
		}
		for (int c = 0; c < 3; ++c) {
			fixed[CUR+c] = 2 * target[CUR+c] - \
				((target[RIGHT+c]) + \
				(target[UP+c])) + tmp[c];
		}
	} else if ( (xt == 0) and (yt == (ht-1)) ) {
		// left-down corner
		if (mask[curt-wt] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 - 3*wb + c];
			}
		}
		if (mask[curt+1] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 + 3 + c];
			}
		}
		for (int c = 0; c < 3; ++c) {
			tmp[c] += background[curb*3 + 3*wb + c];
			tmp[c] += background[curb*3 - 3 + c];
		}
		for (int c = 0; c < 3; ++c) {
			fixed[CUR+c] = 2 * target[CUR+c] - \
				((target[LEFT+c]) + \
				(target[DOWN+c])) + tmp[c];
		}
	} else if ( (xt == (wt-1)) and (yt == (ht-1)) ) {
		// right-down corner
		if (mask[curt-wt] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 - 3*wb + c];
			}
		}
		if (mask[curt-1] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 - 3 + c];
			}
		}
		for (int c = 0; c < 3; ++c) {
			tmp[c] += background[curb*3 + 3*wb + c];
			tmp[c] += background[curb*3 + 3 + c];
		}
		for (int c = 0; c < 3; ++c) {
			fixed[CUR+c] = 2 * target[CUR+c] - \
				((target[LEFT+c]) + \
				(target[UP+c])) + tmp[c];
		}
	} else if ( (xt == 0) ) {
		// left side
		if (mask[curt-wt] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 - 3*wb + c];
			}
		}
		if (mask[curt+1] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 + 3 + c];
			}
		}
		if (mask[curt+wt] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 + 3*wb + c];
			}
		}
		for (int c = 0; c < 3; ++c) {
			tmp[c] += background[curb*3 - 3 + c];
		}
		for (int c = 0; c < 3; ++c) {
			fixed[CUR+c] = 3 * target[CUR+c] - \
				((target[RIGHT+c]) + \
				(target[UP+c]) + \
				(target[DOWN+c])) + tmp[c];
		}
	} else if ( (yt == 0) ) {
		// up side
		if (mask[curt-1] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 - 3 + c];
			}
		}
		if (mask[curt+1] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 + 3 + c];
			}
		}
		if (mask[curt+wt] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 + 3*wb + c];
			}
		}
		for (int c = 0; c < 3; ++c) {
			tmp[c] += background[curb*3 - 3*wb + c];
		}
		for (int c = 0; c < 3; ++c) {
			fixed[CUR+c] = 3 * target[CUR+c] - \
				((target[RIGHT+c]) + \
				(target[LEFT+c]) + \
				(target[DOWN+c])) + tmp[c];
		}
	} else if ( (xt == (wt-1)) ) {
		// right side
		if (mask[curt-wt] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 - 3*wb + c];
			}
		}
		if (mask[curt-1] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 - 3 + c];
			}
		}
		if (mask[curt+wt] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 + 3*wb + c];
			}
		}
		for (int c = 0; c < 3; ++c) {
			tmp[c] += background[curb*3 + 3 + c];
		}
		for (int c = 0; c < 3; ++c) {
			fixed[CUR+c] = 3 * target[CUR+c] - \
				((target[LEFT+c]) + \
				(target[UP+c]) + \
				(target[DOWN+c])) + tmp[c];
		}
	} else if ( (yt == (ht-1)) ) {
		// down side
		if (mask[curt-wt] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 - 3*wb + c];
			}
		}
		if (mask[curt+1] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 + 3 + c];
			}
		}
		if (mask[curt-1] < 127.0f) {
			for (int c = 0; c < 3; ++c) {
				tmp[c] += background[curb*3 - 3 + c];
			}
		}
		for (int c = 0; c < 3; ++c) {
			tmp[c] += background[curb*3 + 3*wb + c];
		}
		for (int c = 0; c < 3; ++c) {
			fixed[CUR+c] = 3 * target[CUR+c] - \
				((target[RIGHT+c]) + \
				(target[UP+c]) + \
				(target[LEFT+c])) + tmp[c];
		}
	}
}

__global__ void PoissonImageCloningIteration (
	const float *fixed,
	const float *mask,
	float *input,
	float *output,
	int wt, int ht
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;

	if ( xt < wt and yt < ht and mask[curt] > 127.0f ) {
		for (int c = 0; c < 3; ++c) {
			output[CUR+c] = 0;
		}
		if( xt > 0 and mask[curt-1] > 127.0f ){
			for (int c = 0; c < 3; ++c) {
				output[CUR+c] += input[LEFT+c]/4;
			}
		}
		if( xt < wt-1 and mask[curt+1] > 127.0f ){
			for (int c = 0; c < 3; ++c) {
				output[CUR+c] += input[RIGHT+c]/4;
			}
		}
		if( yt > 0 and mask[curt-wt] > 127.0f ){
			for (int c = 0; c < 3; ++c) {
				output[CUR+c] += input[UP+c]/4;
			}
		}
		if( yt < ht-1 and mask[curt+wt] > 127.0f ){
			for (int c = 0; c < 3; ++c) {
				output[CUR+c] += input[DOWN+c]/4;
			}
		}
		for (int c = 0; c < 3; ++c) {
			output[CUR+c] += fixed[CUR+c]/4;
		}
	}
	
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	// set up
	float *fixed, *buf1, *buf2;
	cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

	// initialize the iteration
	dim3 gdim(CeilDiv(wt,32), CeilDiv(ht,16)), bdim(32,16);
	CalculateFixed<<<gdim, bdim>>>(
		background, target, mask, fixed,
		wb, hb, wt, ht, oy, ox
	);
	cudaMemcpy(buf1, target, sizeof(float)*3*wt*ht, cudaMemcpyDeviceToDevice);

	// iterate
	for (int i = 0; i < 10000; ++i) {
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf1, buf2, wt, ht
		);
		PoissonImageCloningIteration<<<gdim, bdim>>>(
			fixed, mask, buf2, buf1, wt, ht
		);
	}

	// copy the image back
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<gdim, bdim>>>(
		background, buf1, mask, output,
		wb, hb, wt, ht, oy, ox
	);
	// clean up
	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);
}
