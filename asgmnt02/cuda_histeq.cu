#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <math.h>

extern "C" {
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
}

#define BLOCK_SIZE 256
#define NUM_BINS 256

__global__ void rgb2yuv_kernel(unsigned char* input, float* Y, float* U, float* V, 
                             int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    unsigned char R = input[idx * channels];
    unsigned char G = input[idx * channels + 1];
    unsigned char B = input[idx * channels + 2];
    
    // RGB to YUV conversion
    Y[idx] = 0.299f * R + 0.587f * G + 0.114f * B;
    U[idx] = (-0.168736f * R) + (-0.331264f * G) + 0.5f * B + 128.0f;
    V[idx] = 0.5f * R + (-0.418688f * G) + (-0.081312f * B + 128.0f);
}

__global__ void compute_histogram_kernel(float* Y, int* hist, int total_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;
    
    int bin = min((int)(Y[idx]), NUM_BINS-1);
    atomicAdd(&hist[bin], 1);
}

__global__ void compute_cdf_kernel(int* hist, int* cdf, int total_pixels) {
    // Sequential CDF computation in single thread
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cdf[0] = hist[0];
        for(int i = 1; i < NUM_BINS; i++) {
            cdf[i] = cdf[i-1] + hist[i];
        }
    }
}

__global__ void apply_equalization_kernel(unsigned char* output, float* Y, float* U, float* V, 
                                        int* cdf, int min_cdf, int total_pixels, 
                                        int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    
    // Compute new Y value
    int bin = min((int)Y[idx], NUM_BINS-1);
    float scale = 255.0f / (total_pixels - min_cdf);
    unsigned char Y_new = (cdf[bin] - min_cdf) * scale;
    
    // YUV to RGB conversion
    float U_adj = U[idx] - 128.0f;
    float V_adj = V[idx] - 128.0f;
    
    float R = Y_new + 1.402f * V_adj;
    float G = Y_new - 0.344136f * U_adj - 0.714136f * V_adj;
    float B = Y_new + 1.772f * U_adj;
    
    output[idx * channels] = (unsigned char)fminf(fmaxf(R, 0.0f), 255.0f);
    output[idx * channels + 1] = (unsigned char)fminf(fmaxf(G, 0.0f), 255.0f);
    output[idx * channels + 2] = (unsigned char)fminf(fmaxf(B, 0.0f), 255.0f);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s <input> <output>\n", argv[0]);
        return -1;
    }

    // Load image
    int width, height, channels;
    unsigned char* h_input = stbi_load(argv[1], &width, &height, &channels, 3);
    if (!h_input) {
        fprintf(stderr, "Error loading image\n");
        return -1;
    }
    int total_pixels = width * height;

    // Allocate device memory
    unsigned char *d_input, *d_output;
    float *d_Y, *d_U, *d_V;
    int *d_hist, *d_cdf;
    
    checkCudaErrors(cudaMalloc(&d_input, total_pixels * channels * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&d_output, total_pixels * channels * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&d_Y, total_pixels * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_U, total_pixels * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_V, total_pixels * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_hist, NUM_BINS * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_cdf, NUM_BINS * sizeof(int)));

    // Copy input to device
    checkCudaErrors(cudaMemcpy(d_input, h_input, total_pixels * channels * sizeof(unsigned char), 
               cudaMemcpyHostToDevice));

    // Setup timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Step 1: RGB to YUV conversion
    dim3 blockDim(16, 16);
    dim3 gridDim((width + 15)/16, (height + 15)/16);
    rgb2yuv_kernel<<<gridDim, blockDim>>>(d_input, d_Y, d_U, d_V, width, height, channels);

    // Step 2: Compute histogram
    cudaMemset(d_hist, 0, NUM_BINS * sizeof(int));
    compute_histogram_kernel<<<(total_pixels + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        d_Y, d_hist, total_pixels);

    // Step 3: Compute CDF
    compute_cdf_kernel<<<1, 1>>>(d_hist, d_cdf, total_pixels);

    // Find min_cdf
    int h_cdf[NUM_BINS];
    cudaMemcpy(h_cdf, d_cdf, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);
    int min_cdf = 0;
    for(int i = 0; i < NUM_BINS; i++) {
        if(h_cdf[i] > 0) {
            min_cdf = h_cdf[i];
            break;
        }
    }

    // Step 4: Apply equalization
    apply_equalization_kernel<<<gridDim, blockDim>>>(d_output, d_Y, d_U, d_V, d_cdf,
                                                   min_cdf, total_pixels, 
                                                   width, height, channels);

    // Copy result back
    cudaMemcpy(h_input, d_output, total_pixels * channels * sizeof(unsigned char), 
              cudaMemcpyDeviceToHost);

    // Record time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Save output
    const char* ext = strrchr(argv[2], '.');
    if (!ext) {
        fprintf(stderr, "Missing file extension\n");
        return -1;
    }
    ext++;
    
    if (strcasecmp(ext, "png") == 0) {
        stbi_write_png(argv[2], width, height, channels, h_input, width * channels);
    } else if (strcasecmp(ext, "jpg") == 0 || strcasecmp(ext, "jpeg") == 0) {
        stbi_write_jpg(argv[2], width, height, channels, h_input, 100);
    } else if (strcasecmp(ext, "bmp") == 0) {
        stbi_write_bmp(argv[2], width, height, channels, h_input);
    } else {
        fprintf(stderr, "Unsupported format: %s\n", ext);
    }

    // Cleanup
    cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_Y); cudaFree(d_U); cudaFree(d_V);
    cudaFree(d_hist); cudaFree(d_cdf);
    stbi_image_free(h_input);

    printf("DIMS %d %d TIME %.3f\n", width, height, milliseconds);
    return 0;
}