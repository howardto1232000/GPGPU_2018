#include "lab1.h"
#include <curand.h>
#include <curand_kernel.h>

static const unsigned W = 1000;
static const unsigned H = 550;
static const unsigned NFRAME = 1600;
static int current_frame = 0;
static bool tick = false;

__global__ void draw(uint8_t *, int, int, int, int, int);
__global__ void uu(uint8_t *, int, int, int, int);
__global__ void vv(uint8_t *, int, int, int, int);
__global__ void walld(uint8_t *, int, int);
__global__ void wallu(uint8_t *, int, int);
__global__ void wallr(uint8_t *, int, int);
__global__ void walll(uint8_t *, int, int);

struct Lab1VideoGenerator::Impl {
	// color
	int t = 255;
	int u = 128;
	int v = 128;
	// coordinate of circle
	int coordx = 101;
	int coordy = 101;
	// radius of circle
	int r = 5;
	// direction of cycle
	int dirx = 1;
	int diry = 1;

	int rr = 1;
	// spped of circle
	int speed = 1;

	// wall
	int up = 0;
	int dn = 0;
	int lf = 0;
	int rt = 0;

};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 240/1 = 240
	info.fps_n = 150;
	info.fps_d = 1;
};

void Lab1VideoGenerator::Generate(uint8_t *yuv) {
	//cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);

	//set background
	cudaMemset(yuv+W*H, 128, W*H/4);
	cudaMemset(yuv+W*H+W*H/4, 128, W*H/4);
	
	//(impl->t) = rand() % 255;
	//cudaMemset(yuv, 0, W*H);
	//cudaMemset(yuv+W*H, 0, W*H/2);
	//draw<<<(W*H+W*H/2+479)/480, 480>>>(yuv, (impl->t));

	// draw ball
	draw<<<(W*H+31)/32, 32>>>(yuv, (impl->t), (impl->coordx), (impl->coordy), (impl->r), (impl->t));
	
	// draw color
	uu<<<(W*H/2+31)/32, 32>>>(yuv+W*H, (impl->coordx), (impl->coordy), (impl->r), (impl->u));
	vv<<<(W*H/2+31)/32, 32>>>(yuv+W*H+W*H/4, (impl->coordx), (impl->coordy), (impl->r), (impl->v));

	// ball action
	if (current_frame <= (1600-160)) {
		// ball movement define
		if ((impl->coordx+impl->r < W-1) && (impl->coordy+impl->r < H-1) && 
			(impl->coordx-impl->r > 0) && (impl->coordy-impl->r > 0)) {
		} else if ((impl->coordx+impl->r >= W-1) && (impl->coordy+impl->r >= H-1)) {
			(impl->dirx) = -1*(impl->dirx);
			(impl->diry) = -1*(impl->diry);
			(impl->speed) = ((impl->speed) > 3) ? 3 : ((impl->speed)+1);
			(impl->t) = rand() % 255;
			(impl->u) = rand() % 255;
			(impl->v) = rand() % 255;
			//(impl->r) -= 2;
		} else if ((impl->coordx+impl->r >= W-1)) {
			// hit wall right
			(impl->dirx) = -1*(impl->dirx);
			(impl->speed) = ((impl->speed) > 3) ? 3 : ((impl->speed)+1);
			(impl->t) = rand() % 255;
			(impl->u) = rand() % 255;
			(impl->v) = rand() % 255;
			//(impl->r) -= 2;
            (impl->rt) = 30;
		} else if ((impl->coordy+impl->r >= H-1)){
			// hit wall bottom
			(impl->diry) = -1*(impl->diry);
			(impl->speed) = ((impl->speed) > 3) ? 3 : ((impl->speed)+1);
			(impl->t) = rand() % 255;
			(impl->u) = rand() % 255;
			(impl->v) = rand() % 255;
			//(impl->r) -= 2;
			(impl->dn) = 30;
			//wally<<<(W*H+31)/32, 32>>>(yuv, 0);
		} else if ((impl->coordx-impl->r <= 0) && (impl->coordy-impl->r <= 0)){
			(impl->dirx) = -1*(impl->dirx);
			(impl->diry) = -1*(impl->diry);
			(impl->speed) = ((impl->speed) > 3) ? 3 : ((impl->speed)+1);
			(impl->t) = rand() % 255;
			(impl->u) = rand() % 255;
			(impl->v) = rand() % 255;
			//(impl->r) -= 2;
		} else if ((impl->coordy-impl->r <= 0)){
			// hit wall up
			(impl->diry) = -1*(impl->diry);
			(impl->speed) = ((impl->speed) > 3) ? 3 : ((impl->speed)+1);
			(impl->t) = rand() % 255;
			(impl->u) = rand() % 255;
			(impl->v) = rand() % 255;
			//(impl->r) -= 2;
            (impl->up) = 30;
		} else if ((impl->coordx-impl->r <= 0)) {
			// hit wall left
			(impl->dirx) = -1*(impl->dirx);
			(impl->speed) = ((impl->speed) > 3) ? 3 : ((impl->speed)+1);
			(impl->t) = rand() % 255;
			(impl->u) = rand() % 255;
			(impl->v) = rand() % 255;
			//(impl->r) -= 2;
            (impl->lf) = 30;
		}

        if ((impl->dn) >= 0) {
            int col = (200-(20-(impl->dn))*30) > 0 ? (180-(20-(impl->dn))*5) : 0;
            walld<<<(W*H+31)/32, 32>>>(yuv, 0, col);
            --(impl->dn);
        }

        if ((impl->up) >= 0) {
            int col = (200-(20-(impl->up))*30) > 0 ? (180-(20-(impl->up))*5) : 0;
            wallu<<<(W*H+31)/32, 32>>>(yuv, 0, col);
            --(impl->up);
        }

        if ((impl->rt) >= 0) {
            int col = (200-(20-(impl->rt))*30) > 0 ? (180-(20-(impl->rt))*5) : 0;
            wallr<<<(W*H+31)/32, 32>>>(yuv, 0, col);
            --(impl->rt);
        }

        if ((impl->lf) >= 0) {
            int col = (200-(20-(impl->lf))*30) > 0 ? (180-(20-(impl->lf))*5) : 0;
            walll<<<(W*H+31)/32, 32>>>(yuv, 0, col);
            --(impl->lf);
        }

		(impl->coordx) = ((impl->coordx) + (impl->dirx)*(impl->speed)) % W;
		(impl->coordy) = ((impl->coordy) + (impl->diry)*(impl->speed)) % H;

		if ((current_frame % 20) == 0) {
			(impl->r) += 1;
		}

	} else {
		if (tick == false) {
			(impl->t) -= 50;
			(impl->u) -= 10;
			(impl->v) -= 10;
			tick = true;
		}
		(impl->coordx) = (impl->coordx) + ((W/2-(impl->coordx))/40);
		(impl->coordy) = (impl->coordy) + ((H/2-(impl->coordy))/40);
		if (current_frame > 1600-40) {
			(impl->r) += 4;
		} else if (current_frame > 1600-80){
			(impl->r) += 3;
		} else if (current_frame > 1600-120){
			(impl->r) += 2;
		} else {
			(impl->r) += 1;
		}
		int tmp = (impl->t);
		for (int i = 0; i < (impl->r); i += 2) {
			int color = (tmp + i) < 255 ? (tmp + i) : 255;
			draw<<<(W*H+31)/32, 32>>>(yuv, 0, (impl->coordx), (impl->coordy), ((impl->r)-i), color);
		}
	}
	
	++current_frame;
	//printf("%d \n", current_frame);

}

// perform animation
__global__
void draw(uint8_t *yuv, int seed, int x, int y, int r, int yy) {
	//curandState_t state;
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int tidx = tid % W;
	const int tidy = tid / W;
	//curand_init(seed, 0, 0, &state);
	//yuv[tid] = (uint8_t) curand(&state) % 255;
	if ((tidx-x)*(tidx-x) + (tidy-y)*(tidy-y) <= r*r) {
		yuv[tidy*W + tidx] = (uint8_t)yy;
	} else if (seed == 0) {
	} else {
		yuv[tidy*W + tidx] = 0;
	}
}

// change color
__global__
void uu(uint8_t *u, int x, int y, int r, int col) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int tidx = tid % (W/2);
	const int tidy = tid / (W/2);
	if (((2*tidx-x)*(2*tidx-x))+((2*tidy-y)*(2*tidy-y)) <= r*r)
		u[tid] = (uint8_t)col;
}

__global__
void vv(uint8_t *v, int x, int y, int r, int col) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int tidx = tid % (W/2);
	const int tidy = tid / (W/2);
	if (((2*tidx-x)*(2*tidx-x))+((2*tidy-y)*(2*tidy-y)) <= r*r)
		v[tid] = (uint8_t)col;
}

// 0=up, 1=down, 2=left, 3=right
__global__
void walld(uint8_t *y, int dir, int color) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int tidx = tid % W;
	const int tidy = tid / W;
	if ((dir == 0) && (tidy > (H-5))) {
		y[tid] = (uint8_t) color;
	}
}

__global__
void wallu(uint8_t *y, int dir, int color) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int tidx = tid % W;
	const int tidy = tid / W;
	if ((dir == 0) && (tidy < 5)) {
		y[tid] = (uint8_t) color;
	}
}

__global__
void wallr(uint8_t *y, int dir, int color) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int tidx = tid % W;
	const int tidy = tid / W;
	if ((dir == 0) && (tidx > (W-5))) {
		y[tid] = (uint8_t) color;
	}
}

__global__
void walll(uint8_t *y, int dir, int color) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	const int tidx = tid % W;
	const int tidy = tid / W;
	if ((dir == 0) && (tidx < 5)) {
		y[tid] = (uint8_t) color;
	}
}