#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> // For memcpy
#include <cuda_runtime.h>

// --- CUDA Error Checking Macro ---
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


// --- Shared Memory Tile Dimensions ---
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define TILE_DIM_X (BLOCK_SIZE_X + 2)
#define TILE_DIM_Y (BLOCK_SIZE_Y + 2)

// --- Kernel (Signature unchanged, implementation unchanged) ---
__global__ void gray_scott_kernel(const double* __restrict__ U_in,
                                  const double* __restrict__ V_in,
                                  double* __restrict__ U_out,
                                  double* __restrict__ V_out,
                                  int N, double Du, double Dv, double F, double k, double dt)
{
    // --- Shared memory loading and computation remain the same ---
    // Shared memory tiles for U and V
    __shared__ double tile_U[TILE_DIM_Y][TILE_DIM_X];
    __shared__ double tile_V[TILE_DIM_Y][TILE_DIM_X];

    // Global grid coordinates (i, j) for this thread
    const int j_global = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_global = blockIdx.y * blockDim.y + threadIdx.y;

    // Local thread indices within the block (tx, ty)
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Local indices within the shared memory tile (including halo)
    const int tile_x = tx + 1; // Offset by 1 for halo
    const int tile_y = ty + 1; // Offset by 1 for halo

    // --- Load data into shared memory (including halo) ---
    // (Exact halo loading logic depends on previous implementation, keep it consistent)
     // Each thread loads one central element into the tile
    if (i_global < N && j_global < N) {
        tile_U[tile_y][tile_x] = U_in[i_global * N + j_global];
        tile_V[tile_y][tile_x] = V_in[i_global * N + j_global];
    } else {
         tile_U[tile_y][tile_x] = 0.0; tile_V[tile_y][tile_x] = 0.0;
    }
    // Handle halo regions (threads at the block edges load extra data)
    // Using periodic boundary conditions (wrapping)
    // Top halo row (ty == 0)
    if (ty == 0) {
        int src_i = (i_global - 1 + N) % N;
        if (j_global < N) {
             tile_U[0][tile_x] = U_in[src_i * N + j_global];
             tile_V[0][tile_x] = V_in[src_i * N + j_global];
        } else { tile_U[0][tile_x] = 0.0; tile_V[0][tile_x] = 0.0; }
        if (tx == 0) {
            int src_j = (j_global - 1 + N) % N;
            tile_U[0][0] = U_in[src_i * N + src_j];
            tile_V[0][0] = V_in[src_i * N + src_j];
        }
        if (tx == blockDim.x - 1) {
            int src_j = (j_global + 1) % N;
             if (src_j < N) {
                 tile_U[0][TILE_DIM_X - 1] = U_in[src_i * N + src_j];
                 tile_V[0][TILE_DIM_X - 1] = V_in[src_i * N + src_j];
            } else { tile_U[0][TILE_DIM_X - 1] = 0.0; tile_V[0][TILE_DIM_X - 1] = 0.0; }
        }
    }
    // Bottom halo row (ty == blockDim.y - 1)
    if (ty == blockDim.y - 1) {
        int src_i = (i_global + 1) % N;
         if (src_i < N && j_global < N) {
             tile_U[TILE_DIM_Y - 1][tile_x] = U_in[src_i * N + j_global];
             tile_V[TILE_DIM_Y - 1][tile_x] = V_in[src_i * N + j_global];
        } else { tile_U[TILE_DIM_Y - 1][tile_x] = 0.0; tile_V[TILE_DIM_Y - 1][tile_x] = 0.0; }
        if (tx == 0) {
            int src_j = (j_global - 1 + N) % N;
            if (src_i < N) {
                tile_U[TILE_DIM_Y - 1][0] = U_in[src_i * N + src_j];
                tile_V[TILE_DIM_Y - 1][0] = V_in[src_i * N + src_j];
            } else { tile_U[TILE_DIM_Y - 1][0] = 0.0; tile_V[TILE_DIM_Y - 1][0] = 0.0; }
        }
        if (tx == blockDim.x - 1) {
            int src_j = (j_global + 1) % N;
             if (src_i < N && src_j < N) {
                 tile_U[TILE_DIM_Y - 1][TILE_DIM_X - 1] = U_in[src_i * N + src_j];
                 tile_V[TILE_DIM_Y - 1][TILE_DIM_X - 1] = V_in[src_i * N + src_j];
            } else { tile_U[TILE_DIM_Y - 1][TILE_DIM_X - 1] = 0.0; tile_V[TILE_DIM_Y - 1][TILE_DIM_X - 1] = 0.0; }
        }
    }
    // Left halo column (tx == 0)
    if (tx == 0) {
        int src_j = (j_global - 1 + N) % N;
        if (i_global < N) {
             tile_U[tile_y][0] = U_in[i_global * N + src_j];
             tile_V[tile_y][0] = V_in[i_global * N + src_j];
        } else { tile_U[tile_y][0] = 0.0; tile_V[tile_y][0] = 0.0; }
    }
    // Right halo column (tx == blockDim.x - 1)
    if (tx == blockDim.x - 1) {
        int src_j = (j_global + 1) % N;
         if (i_global < N && src_j < N) {
             tile_U[tile_y][TILE_DIM_X - 1] = U_in[i_global * N + src_j];
             tile_V[tile_y][TILE_DIM_X - 1] = V_in[i_global * N + src_j];
        } else { tile_U[tile_y][TILE_DIM_X - 1] = 0.0; tile_V[tile_y][TILE_DIM_X - 1] = 0.0; }
    }

    __syncthreads(); // Synchronize threads

    // --- Perform computation using shared memory ---
    if (i_global < N && j_global < N) {
        double u_ijk = tile_U[tile_y][tile_x];
        double v_ijk = tile_V[tile_y][tile_x];

        double laplace_U = tile_U[tile_y + 1][tile_x] + tile_U[tile_y - 1][tile_x] +
                           tile_U[tile_y][tile_x + 1] + tile_U[tile_y][tile_x - 1] -
                           4.0 * u_ijk;

        double laplace_V = tile_V[tile_y + 1][tile_x] + tile_V[tile_y - 1][tile_x] +
                           tile_V[tile_y][tile_x + 1] + tile_V[tile_y][tile_x - 1] -
                           4.0 * v_ijk;

        double reaction = u_ijk * v_ijk * v_ijk;

        U_out[i_global * N + j_global] = u_ijk + dt * (Du * laplace_U - reaction + F * (1.0 - u_ijk));
        V_out[i_global * N + j_global] = v_ijk + dt * (Dv * laplace_V + reaction - (F + k) * v_ijk);
    }
}


// --- PGM Saving Function (same as sequential version, adapted for CUDA context) ---
// *** MODIFIED: Include F and k in filename ***
void save_pgm(const double* grid, int N, int steps, double F, double k, const char* prefix) {
    char filename[150];
    snprintf(filename, sizeof(filename), "%s_N%d_steps%d_F%.3f_k%.3f_V.pgm", prefix, N, steps, F, k);

     FILE* f = fopen(filename, "w");
    if (!f) {
        perror("Failed to open output file");
        fprintf(stderr, "Filename attempted: %s\n", filename);
        return;
    }
    fprintf(f, "P2\n%d %d\n255\n", N, N);
    double min_val = grid[0], max_val = grid[0];
    for (int i = 0; i < N * N; ++i) {
        if (grid[i] < min_val) min_val = grid[i];
        if (grid[i] > max_val) max_val = grid[i];
    }
    if (max_val <= min_val) max_val = min_val + 1e-6;

    int count = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double val = grid[i * N + j];
            int gray_val = (int)(255.0 * (val - min_val) / (max_val - min_val));
            if (gray_val < 0) gray_val = 0;
            if (gray_val > 255) gray_val = 255;
            fprintf(f, "%d", gray_val);
            count++;
            if (count % 16 == 0 || j == N - 1) {
                 fprintf(f, "\n");
            } else {
                 fprintf(f, " ");
            }
        }
    }
    fclose(f);
    printf("INFO: Saved V grid to %s\n", filename);
}

// --- Main Function ---
int main(int argc, char* argv[]) {
    // *** MODIFIED: Expect 5 arguments: N, steps, F, k ***
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <N> <steps> <F> <k>\n", argv[0]);
        fprintf(stderr, "  N: grid size (NxN)\n");
        fprintf(stderr, "  steps: number of simulation steps\n");
        fprintf(stderr, "  F: feed rate (e.g., 0.060)\n");
        fprintf(stderr, "  k: kill rate (e.g., 0.062)\n");
        return 1;
    }

    int N = atoi(argv[1]);
    int steps = atoi(argv[2]);
    // *** MODIFIED: Parse F and k from command line ***
    double F = atof(argv[3]); // Feed rate
    double k = atof(argv[4]); // Kill rate

    if (N <= 0 || steps <= 0 || F <= 0 || k <= 0) {
        fprintf(stderr, "Error: N, steps, F, and k must be positive numbers.\n");
        return 1;
    }
    printf("INFO: Running CUDA Gray-Scott N=%d, Steps=%d, F=%.4f, k=%.4f\n", N, steps, F, k);


    // Other simulation parameters (fixed)
    const double dt = 1.0;
    const double Du = 0.16;
    const double Dv = 0.08;

    // --- Host Memory Allocation ---
    size_t gridSize = (size_t)N * N;
    size_t bytes = gridSize * sizeof(double);
    // We only need host V buffer for the final result.
    // Host U buffer needed just for initialization before copying to device.
    double* h_U_init = (double*)malloc(bytes);
    double* h_V      = (double*)malloc(bytes);
    if (!h_U_init || !h_V) {
        perror("Failed to allocate host memory");
        free(h_U_init); free(h_V);
        return 1;
    }
    printf("INFO: Allocated %.2f MB host memory.\n", 2.0 * bytes / (1024.0 * 1024.0));

    // --- Initialize Grids on Host (using h_U_init and h_V) ---
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_U_init[i * N + j] = 1.0;
            h_V[i * N + j] = 0.0; // Initialize V here too
        }
    }
    int size = N / 4;
    int start = N / 2 - size / 2;
    int end = start + size;
     if (start < 0) start = 0;
     if (end > N) end = N;
    for (int i = start; i < end; ++i) {
        for (int j = start; j < end; ++j) {
             if (i >= 0 && i < N && j >= 0 && j < N) {
                h_U_init[i * N + j] = 0.50;
                h_V[i * N + j] = 0.25;
             }
        }
    }


    // --- Device Memory Allocation ---
    double *d_U, *d_V, *d_U_new, *d_V_new;
    CHECK_CUDA(cudaMalloc(&d_U, bytes));
    CHECK_CUDA(cudaMalloc(&d_V, bytes));
    CHECK_CUDA(cudaMalloc(&d_U_new, bytes));
    CHECK_CUDA(cudaMalloc(&d_V_new, bytes));
    printf("INFO: Allocated %.2f MB device memory.\n", 4.0 * bytes / (1024.0 * 1024.0));

    // --- Time Measurement Start (includes H->D copy, kernel loop, D->H copy) ---
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // --- Copy Initial Data Host -> Device ---
    // Copy from the initialized host buffers
    CHECK_CUDA(cudaMemcpy(d_U, h_U_init, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));
    // Free the temporary host U buffer now
    free(h_U_init); h_U_init = NULL;


    // --- Kernel Launch Configuration ---
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    printf("INFO: Launching kernel with gridDim=(%d, %d), blockDim=(%d, %d)\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // --- Simulation Loop ---
    for (int step = 0; step < steps; ++step) {
        // Launch the kernel, passing the parsed F and k
        gray_scott_kernel<<<gridDim, blockDim>>>(d_U, d_V, d_U_new, d_V_new, N, Du, Dv, F, k, dt);
        CHECK_CUDA(cudaGetLastError()); // Check for launch errors

        // Swap device pointers
        double* temp_U = d_U; d_U = d_U_new; d_U_new = temp_U;
        double* temp_V = d_V; d_V = d_V_new; d_V_new = temp_V;
    }

    // --- Synchronize device ---
    CHECK_CUDA(cudaDeviceSynchronize());

    // --- Copy Final Result Device -> Host ---
    // Result is in d_V, copy it to h_V
    CHECK_CUDA(cudaMemcpy(h_V, d_V, bytes, cudaMemcpyDeviceToHost));

    // --- Time Measurement End ---
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                          (end_time.tv_nsec - start_time.tv_nsec) / 1e9;


    // --- Print results ---
    printf("DIMS %dx%d TIME %f\n", N, N, elapsed_time);


    // --- Save final V grid ---
    // *** MODIFIED: Pass F and k to save_pgm, use "gray_scott_cuda" as prefix ***
    save_pgm(h_V, N, steps, F, k, "gray_scott_cuda");

    // --- Cleanup ---
    CHECK_CUDA(cudaFree(d_U));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_U_new));
    CHECK_CUDA(cudaFree(d_V_new));
    free(h_V);

    printf("INFO: CUDA simulation finished successfully.\n");

    return 0;
}