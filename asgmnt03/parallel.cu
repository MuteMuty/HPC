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
// Choose block size (e.g., 16x16 = 256 threads, or 32x32 = 1024 threads)
// Must be <= 1024 total threads per block
// Using 16x16 is often a safe starting point
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
// Shared memory tile needs halo of 1 cell on each side for 5-point stencil
#define TILE_DIM_X (BLOCK_SIZE_X + 2)
#define TILE_DIM_Y (BLOCK_SIZE_Y + 2)

// --- Kernel ---
__global__ void gray_scott_kernel(const double* __restrict__ U_in,
                                  const double* __restrict__ V_in,
                                  double* __restrict__ U_out,
                                  double* __restrict__ V_out,
                                  int N, double Du, double Dv, double F, double k, double dt)
{
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
    // Each thread loads one central element into the tile
    if (i_global < N && j_global < N) {
        tile_U[tile_y][tile_x] = U_in[i_global * N + j_global];
        tile_V[tile_y][tile_x] = V_in[i_global * N + j_global];
    } else {
        // Pad with some value if outside grid (though bounds check later should prevent use)
         tile_U[tile_y][tile_x] = 0.0;
         tile_V[tile_y][tile_x] = 0.0;
    }

    // Handle halo regions (threads at the block edges load extra data)
    // Using periodic boundary conditions (wrapping)

    // Top halo row (ty == 0)
    if (ty == 0) {
        int src_i = (i_global - 1 + N) % N;
        if (j_global < N) { // Check column index
             tile_U[0][tile_x] = U_in[src_i * N + j_global];
             tile_V[0][tile_x] = V_in[src_i * N + j_global];
        } else { tile_U[0][tile_x] = 0.0; tile_V[0][tile_x] = 0.0; }
        // Top-left corner (also tx == 0)
        if (tx == 0) {
            int src_j = (j_global - 1 + N) % N;
            tile_U[0][0] = U_in[src_i * N + src_j];
            tile_V[0][0] = V_in[src_i * N + src_j];
        }
        // Top-right corner (also tx == blockDim.x - 1)
        if (tx == blockDim.x - 1) {
            int src_j = (j_global + 1) % N;
            if (src_j < N) { // Check wrapped column index
                 tile_U[0][TILE_DIM_X - 1] = U_in[src_i * N + src_j];
                 tile_V[0][TILE_DIM_X - 1] = V_in[src_i * N + src_j];
            } else { tile_U[0][TILE_DIM_X - 1] = 0.0; tile_V[0][TILE_DIM_X - 1] = 0.0; }
        }
    }

    // Bottom halo row (ty == blockDim.y - 1)
    if (ty == blockDim.y - 1) {
        int src_i = (i_global + 1) % N;
         if (src_i < N && j_global < N) { // Check indices
             tile_U[TILE_DIM_Y - 1][tile_x] = U_in[src_i * N + j_global];
             tile_V[TILE_DIM_Y - 1][tile_x] = V_in[src_i * N + j_global];
        } else { tile_U[TILE_DIM_Y - 1][tile_x] = 0.0; tile_V[TILE_DIM_Y - 1][tile_x] = 0.0; }
        // Bottom-left corner (also tx == 0)
        if (tx == 0) {
            int src_j = (j_global - 1 + N) % N;
            if (src_i < N) { // Check row index
                tile_U[TILE_DIM_Y - 1][0] = U_in[src_i * N + src_j];
                tile_V[TILE_DIM_Y - 1][0] = V_in[src_i * N + src_j];
            } else { tile_U[TILE_DIM_Y - 1][0] = 0.0; tile_V[TILE_DIM_Y - 1][0] = 0.0; }
        }
        // Bottom-right corner (also tx == blockDim.x - 1)
        if (tx == blockDim.x - 1) {
            int src_j = (j_global + 1) % N;
             if (src_i < N && src_j < N) { // Check indices
                 tile_U[TILE_DIM_Y - 1][TILE_DIM_X - 1] = U_in[src_i * N + src_j];
                 tile_V[TILE_DIM_Y - 1][TILE_DIM_X - 1] = V_in[src_i * N + src_j];
            } else { tile_U[TILE_DIM_Y - 1][TILE_DIM_X - 1] = 0.0; tile_V[TILE_DIM_Y - 1][TILE_DIM_X - 1] = 0.0; }
        }
    }

    // Left halo column (tx == 0)
    if (tx == 0) {
        int src_j = (j_global - 1 + N) % N;
        if (i_global < N) { // Check row index
             tile_U[tile_y][0] = U_in[i_global * N + src_j];
             tile_V[tile_y][0] = V_in[i_global * N + src_j];
        } else { tile_U[tile_y][0] = 0.0; tile_V[tile_y][0] = 0.0; }
    }

    // Right halo column (tx == blockDim.x - 1)
    if (tx == blockDim.x - 1) {
        int src_j = (j_global + 1) % N;
         if (i_global < N && src_j < N) { // Check indices
             tile_U[tile_y][TILE_DIM_X - 1] = U_in[i_global * N + src_j];
             tile_V[tile_y][TILE_DIM_X - 1] = V_in[i_global * N + src_j];
        } else { tile_U[tile_y][TILE_DIM_X - 1] = 0.0; tile_V[tile_y][TILE_DIM_X - 1] = 0.0; }
    }

    // Synchronize threads within the block to ensure shared memory is fully loaded
    __syncthreads();

    // --- Perform computation using shared memory ---
    // Check if the thread is responsible for a valid grid cell
    if (i_global < N && j_global < N) {
        // Get current U and V from the center of the tile
        double u_ijk = tile_U[tile_y][tile_x];
        double v_ijk = tile_V[tile_y][tile_x];

        // Calculate Laplacian using neighbors from shared memory tile
        double laplace_U = tile_U[tile_y + 1][tile_x] + tile_U[tile_y - 1][tile_x] +
                           tile_U[tile_y][tile_x + 1] + tile_U[tile_y][tile_x - 1] -
                           4.0 * u_ijk;

        double laplace_V = tile_V[tile_y + 1][tile_x] + tile_V[tile_y - 1][tile_x] +
                           tile_V[tile_y][tile_x + 1] + tile_V[tile_y][tile_x - 1] -
                           4.0 * v_ijk;

        // Calculate reaction term
        double reaction = u_ijk * v_ijk * v_ijk;

        // Apply Gray-Scott update rules
        U_out[i_global * N + j_global] = u_ijk + dt * (Du * laplace_U - reaction + F * (1.0 - u_ijk));
        V_out[i_global * N + j_global] = v_ijk + dt * (Dv * laplace_V + reaction - (F + k) * v_ijk);

        // Optional clamping (can sometimes help stability)
        // U_out[i_global * N + j_global] = fmax(0.0, fmin(1.0, U_out[i_global * N + j_global]));
        // V_out[i_global * N + j_global] = fmax(0.0, fmin(1.0, V_out[i_global * N + j_global]));
    }
}


// --- PGM Saving Function (same as sequential, operates on host data) ---
void save_pgm(const double* grid, int N, const char* filename) {
     FILE* f = fopen(filename, "w");
    if (!f) {
        perror("Failed to open output file");
        return;
    }
    fprintf(f, "P2\n%d %d\n255\n", N, N);
    double min_val = grid[0], max_val = grid[0];
    for (int i = 0; i < N * N; ++i) {
        if (grid[i] < min_val) min_val = grid[i];
        if (grid[i] > max_val) max_val = grid[i];
    }
    if (max_val <= min_val) max_val = min_val + 1e-6; // Avoid division by zero

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
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <N> <steps>\n", argv[0]);
        fprintf(stderr, "  N: grid size (NxN)\n");
        fprintf(stderr, "  steps: number of simulation steps\n");
        return 1;
    }

    int N = atoi(argv[1]);
    int steps = atoi(argv[2]);

    if (N <= 0 || steps <= 0) {
        fprintf(stderr, "Error: N and steps must be positive integers.\n");
        return 1;
    }
     // Ensure N is reasonably compatible with block sizes if needed, though not strictly required
     if (N % BLOCK_SIZE_X != 0 || N % BLOCK_SIZE_Y != 0) {
         // printf("Warning: Grid size N (%d) is not perfectly divisible by block dimensions (%d, %d).\n",
         //        N, BLOCK_SIZE_X, BLOCK_SIZE_Y);
         // This is fine, the kernel handles boundary checks.
     }


    // Simulation parameters
    const double dt = 1.0;
    const double Du = 0.16;
    const double Dv = 0.08;
    const double F = 0.060;
    const double k = 0.062;

    // --- Host Memory Allocation ---
    size_t gridSize = (size_t)N * N;
    size_t bytes = gridSize * sizeof(double);
    double* h_U = (double*)malloc(bytes);
    double* h_V = (double*)malloc(bytes); // Only h_V needed for final result copy-back
    if (!h_U || !h_V) {
        perror("Failed to allocate host memory");
        free(h_U); free(h_V);
        return 1;
    }
    printf("INFO: Allocated %.2f MB host memory.\n", 2.0 * bytes / (1024.0 * 1024.0));

    // --- Initialize Grids on Host ---
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_U[i * N + j] = 1.0;
            h_V[i * N + j] = 0.0;
        }
    }
    int size = N / 4;
    int start = N / 2 - size / 2;
    int end = start + size; // Non-inclusive end
     if (start < 0) start = 0;
     if (end > N) end = N;
    for (int i = start; i < end; ++i) {
        for (int j = start; j < end; ++j) {
             if (i >= 0 && i < N && j >= 0 && j < N) {
                h_U[i * N + j] = 0.75; // Using 0.75 as in the sequential example comment
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
    CHECK_CUDA(cudaMemcpy(d_U, h_U, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));
    // No need to copy h_U anymore, free it now if memory is tight (optional)
    free(h_U); h_U = NULL;

    // --- Kernel Launch Configuration ---
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    // Calculate grid dimensions to cover the entire N x N grid
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    printf("INFO: Launching kernel with gridDim=(%d, %d), blockDim=(%d, %d)\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // --- Simulation Loop ---
    for (int step = 0; step < steps; ++step) {
        // Launch the kernel
        gray_scott_kernel<<<gridDim, blockDim>>>(d_U, d_V, d_U_new, d_V_new, N, Du, Dv, F, k, dt);

        // Check for kernel launch errors (optional, but good for debugging)
        // CHECK_CUDA(cudaGetLastError());

        // Swap device pointers for next iteration
        double* temp_U = d_U;
        d_U = d_U_new;
        d_U_new = temp_U;

        double* temp_V = d_V;
        d_V = d_V_new;
        d_V_new = temp_V;

        // Optional: Synchronize and check errors periodically if debugging issues
        // if ((step + 1) % 100 == 0) {
        //     CHECK_CUDA(cudaDeviceSynchronize());
        //     printf("Step %d completed\n", step + 1);
        // }
    }

    // --- Synchronize device to ensure all computations are finished ---
    CHECK_CUDA(cudaDeviceSynchronize());

    // --- Copy Final Result Device -> Host ---
    // The final result is in d_V (if steps is even) or d_V_new (if steps is odd)
    // Due to the pointer swap, d_V always points to the *last computed* V grid.
    CHECK_CUDA(cudaMemcpy(h_V, d_V, bytes, cudaMemcpyDeviceToHost));

    // --- Time Measurement End ---
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                          (end_time.tv_nsec - start_time.tv_nsec) / 1e9;


    // --- Print results ---
    printf("DIMS %dx%d TIME %f\n", N, N, elapsed_time);


    // --- Save final V grid ---
    char filename[100];
    snprintf(filename, sizeof(filename), "parallel_N%d_steps%d_V.pgm", N, steps);
    save_pgm(h_V, N, filename);

    // --- Cleanup ---
    CHECK_CUDA(cudaFree(d_U));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_U_new));
    CHECK_CUDA(cudaFree(d_V_new));
    free(h_V); // h_U was potentially freed earlier

    printf("INFO: CUDA simulation finished successfully.\n");

    return 0;
}