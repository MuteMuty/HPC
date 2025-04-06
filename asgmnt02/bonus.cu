#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define NUM_BINS 256
#define SHARED_HIST_SIZE (NUM_BINS + 2)  // Padding for bank conflicts
#define BLOCK_SIZE 256
#define SCAN_BLOCK_SIZE 512

// Dummy kernel for warm-up
__global__ void dummy_kernel() {
    // Do nothing
}

// Kernel to convert RGB to YUV
__global__ void rgb_to_yuv_kernel(unsigned char *d_image, unsigned char *d_Y, float *d_U, float *d_V, int width, int height, int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;
    int idx = i * channels;
    unsigned char R = d_image[idx];
    unsigned char G = d_image[idx + 1];
    unsigned char B = d_image[idx + 2];
    float Y_f = 0.299f * R + 0.587f * G + 0.114f * B;
    d_Y[i] = static_cast<unsigned char>(Y_f + 0.5f);
    d_U[i] = (-0.168736f * R) + (-0.331264f * G) + (0.5f * B) + 128.0f;
    d_V[i] = (0.5f * R) + (-0.418688f * G) + (-0.081312f * B) + 128.0f;
}

// Shared memory histogram with reduction
__global__ void histogram_shared_kernel(unsigned char* d_Y, int* d_hist, int size) {
    __shared__ int s_hist[SHARED_HIST_SIZE];
    for(int i = threadIdx.x; i < SHARED_HIST_SIZE; i += blockDim.x)
        s_hist[i] = 0;
    __syncthreads();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(idx < size) {
        unsigned char val = d_Y[idx];
        atomicAdd(&s_hist[val], 1);
        idx += stride;
    }
    __syncthreads();
    for(int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&d_hist[i], s_hist[i]);
    }
}

// Blelloch work-efficient parallel scan
__device__ void blelloch_scan(int* data, int n) {
    for(int d = 0; d < log2((float)n); d++) {
        int stride = 1 << (d + 1);
        int offset = 1 << d;
        for(int k = threadIdx.x * stride; k < n; k += blockDim.x * stride) {
            int idx = k + stride - 1;
            if(idx < n && idx - offset >= 0) {
                data[idx] += data[idx - offset];
            }
        }
        __syncthreads();
    }
    data[n-1] = 0;
    for(int d = log2((float)n) - 1; d >= 0; d--) {
        int stride = 1 << (d + 1);
        int offset = 1 << d;
        for(int k = threadIdx.x * stride; k < n; k += blockDim.x * stride) {
            int idx = k + offset - 1;
            if(idx < n) {
                int temp = data[idx];
                data[idx] = data[idx + offset];
                data[idx + offset] += temp;
            }
        }
        __syncthreads();
    }
}

__global__ void parallel_scan_kernel(int* d_cdf, int* d_hist) {
    __shared__ int s_data[SCAN_BLOCK_SIZE];
    int idx = threadIdx.x;
    s_data[idx] = (idx < NUM_BINS) ? d_hist[idx] : 0;
    __syncthreads();
    blelloch_scan(s_data, SCAN_BLOCK_SIZE);
    if(idx < NUM_BINS) {
        d_cdf[idx] = s_data[idx];
    }
}

// Kernel to find the minimum non-zero value in CDF
__global__ void find_min_cdf_kernel(int *d_cdf, int *d_min_cdf) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_min_cdf = 0;
        for (int l = 0; l < 256; ++l) {
            if (d_cdf[l] > 0) {
                *d_min_cdf = d_cdf[l];
                break;
            }
        }
    }
}

// Kernel to compute the lookup table (LUT)
__global__ void compute_LUT_kernel(unsigned char *d_LUT, int *d_cdf, int *d_min_cdf, int total_pixels) {
    int l = threadIdx.x;
    if (l >= 256) return;
    int cdf_val = d_cdf[l];
    int min_cdf = *d_min_cdf;
    if (cdf_val == 0) {
        d_LUT[l] = 0;
    } else {
        int denominator = total_pixels - min_cdf;
        if (denominator <= 0) {
            d_LUT[l] = 255;
        } else {
            float scale = 255.0f / denominator;
            float normalized = (cdf_val - min_cdf) * scale;
            normalized = fmaxf(normalized, 0.0f);
            normalized = fminf(normalized, 255.0f);
            d_LUT[l] = static_cast<unsigned char>(normalized + 0.5f);
        }
    }
}

// Kernel to apply LUT to Y channel
__global__ void apply_LUT_kernel(unsigned char *d_Y, unsigned char *d_LUT, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    unsigned char y_val = d_Y[i];
    d_Y[i] = d_LUT[y_val];
}

// Kernel to convert YUV back to RGB
__global__ void yuv_to_rgb_kernel(unsigned char *d_image, unsigned char *d_Y, float *d_U, float *d_V, int width, int height, int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;
    unsigned char Y_val = d_Y[i];
    float U_val = d_U[i] - 128.0f;
    float V_val = d_V[i] - 128.0f;
    float R = Y_val + 1.402f * V_val;
    float G = Y_val - 0.344136f * U_val - 0.714136f * V_val;
    float B = Y_val + 1.772f * U_val;
    R = fmaxf(0.0f, fminf(R, 255.0f));
    G = fmaxf(0.0f, fminf(G, 255.0f));
    B = fmaxf(0.0f, fminf(B, 255.0f));
    int idx = i * channels;
    d_image[idx]     = static_cast<unsigned char>(R + 0.5f);
    d_image[idx + 1] = static_cast<unsigned char>(G + 0.5f);
    d_image[idx + 2] = static_cast<unsigned char>(B + 0.5f);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return -1;
    }
    const char* input_path = argv[1];
    const char* output_path = argv[2];
    int width, height, channels;
    unsigned char* h_image = stbi_load(input_path, &width, &height, &channels, 3);
    if (!h_image) {
        fprintf(stderr, "Error loading image %s\n", input_path);
        return -1;
    }
    printf("Loaded image %s (%dx%d, %d channels)\n", input_path, width, height, channels);
    const size_t image_size = width * height * channels * sizeof(unsigned char);
    const int total_pixels = width * height;

    // Allocate device memory
    unsigned char *d_image, *d_Y, *d_LUT;
    float *d_U, *d_V;
    int *d_histogram, *d_cdf, *d_min_cdf;
    checkCudaErrors(cudaMalloc((void**)&d_image, image_size));
    checkCudaErrors(cudaMalloc((void**)&d_Y, total_pixels * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc((void**)&d_U, total_pixels * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_V, total_pixels * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_histogram, 256 * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_cdf, 256 * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_min_cdf, sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_LUT, 256 * sizeof(unsigned char)));

    // Initialize histogram to zero
    checkCudaErrors(cudaMemset(d_histogram, 0, 256 * sizeof(int)));
    // Copy input image to device
    checkCudaErrors(cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice));

    // Warm-up call
    dummy_kernel<<<1, 1>>>();
    checkCudaErrors(cudaDeviceSynchronize());

    // CUDA events for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));
    
    // Launch RGB to YUV kernel
    dim3 block(256);
    dim3 grid((total_pixels + block.x - 1) / block.x);
    rgb_to_yuv_kernel<<<grid, block>>>(d_image, d_Y, d_U, d_V, width, height, channels);
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Compute histogram using shared memory
    histogram_shared_kernel<<<256, BLOCK_SIZE>>>(d_Y, d_histogram, total_pixels);
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Parallel scan for CDF
    parallel_scan_kernel<<<1, SCAN_BLOCK_SIZE>>>(d_cdf, d_histogram);
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Find min CDF
    find_min_cdf_kernel<<<1, 1>>>(d_cdf, d_min_cdf);
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Compute LUT
    compute_LUT_kernel<<<1, 256>>>(d_LUT, d_cdf, d_min_cdf, total_pixels);
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Apply LUT to Y channel
    apply_LUT_kernel<<<grid, block>>>(d_Y, d_LUT, total_pixels);
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Convert YUV back to RGB
    yuv_to_rgb_kernel<<<grid, block>>>(d_image, d_Y, d_U, d_V, width, height, channels);
    checkCudaErrors(cudaDeviceSynchronize());
    
    checkCudaErrors(cudaMemcpy(h_image, d_image, image_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("DIMS %d %d TIME %.3f\n", width, height, milliseconds);
    
    // Write output image to bonus_results/outputs
    char output_path_full[512];
    snprintf(output_path_full, sizeof(output_path_full), "bonus_results/outputs/%s", output_path);
    const char* ext = strrchr(output_path, '.');
    if (!ext) {
        fprintf(stderr, "Output file must have an extension\n");
        stbi_image_free(h_image);
        return -1;
    }
    ext++;
    int success;
    if (strcasecmp(ext, "png") == 0) {
        success = stbi_write_png(output_path, width, height, channels, h_image, width * channels);
    } else if (strcasecmp(ext, "jpg") == 0 || strcasecmp(ext, "jpeg") == 0) {
        success = stbi_write_jpg(output_path, width, height, channels, h_image, 100);
    } else if (strcasecmp(ext, "bmp") == 0) {
        success = stbi_write_bmp(output_path, width, height, channels, h_image);
    } else {
        fprintf(stderr, "Unsupported format: %s\n", ext);
        success = 0;
    }
    if (!success) {
        fprintf(stderr, "Error writing image to %s\n", output_path);
    }
    
    // Cleanup
    stbi_image_free(h_image);
    checkCudaErrors(cudaFree(d_image));
    checkCudaErrors(cudaFree(d_Y));
    checkCudaErrors(cudaFree(d_U));
    checkCudaErrors(cudaFree(d_V));
    checkCudaErrors(cudaFree(d_histogram));
    checkCudaErrors(cudaFree(d_cdf));
    checkCudaErrors(cudaFree(d_min_cdf));
    checkCudaErrors(cudaFree(d_LUT));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    return 0;
}
