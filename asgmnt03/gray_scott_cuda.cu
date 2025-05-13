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
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 16
// Halo of 1 cell on each side
#define TILE_DIM_X (BLOCK_SIZE_X + 2)
#define TILE_DIM_Y (BLOCK_SIZE_Y + 2)

__global__ void gray_scott_kernel(const double* __restrict__ U_in,
                                  const double* __restrict__ V_in,
                                  double* __restrict__ U_out,
                                  double* __restrict__ V_out,
                                  int N, double Du, double Dv, double F, double k, double dt)
{
    __shared__ double tile_U[TILE_DIM_Y][TILE_DIM_X];
    __shared__ double tile_V[TILE_DIM_Y][TILE_DIM_X];

    // Raw global coordinates of this thread (can be >= N)
    const int j_global_raw = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_global_raw = blockIdx.y * blockDim.y + threadIdx.y;

    // Local thread indices within the block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // --- Load data into shared memory with periodic wrapping ---

    // 1. Each thread loads its "central" point for the tile view.
    // tile[ty+1][tx+1] corresponds to data from (i_global_raw, j_global_raw)
    int r_load = (i_global_raw % N); // Wrap row index
    int c_load = (j_global_raw % N); // Wrap col index
    tile_U[ty + 1][tx + 1] = U_in[r_load * N + c_load];
    tile_V[ty + 1][tx + 1] = V_in[r_load * N + c_load];

    // 2. Threads at the edges of the block load halo regions.
    // Top halo row: tile_U[0][tx+1], loaded by threads with ty=0
    if (ty == 0) {
        r_load = ((i_global_raw - 1 + N) % N); // Row "above" current thread's raw position, wrapped
        c_load = (j_global_raw % N);           // Same column as current thread's raw position, wrapped
        tile_U[0][tx + 1] = U_in[r_load * N + c_load];
        tile_V[0][tx + 1] = V_in[r_load * N + c_load];
    }
    // Bottom halo row: tile_U[TILE_DIM_Y-1][tx+1], loaded by ty=blockDim.y-1
    if (ty == blockDim.y - 1) {
        r_load = ((i_global_raw + 1 + N) % N); // Row "below"
        c_load = (j_global_raw % N);
        tile_U[TILE_DIM_Y - 1][tx + 1] = U_in[r_load * N + c_load];
        tile_V[TILE_DIM_Y - 1][tx + 1] = V_in[r_load * N + c_load];
    }
    // Left halo col: tile_U[ty+1][0], loaded by tx=0
    if (tx == 0) {
        r_load = (i_global_raw % N);
        c_load = ((j_global_raw - 1 + N) % N); // Col "left"
        tile_U[ty + 1][0] = U_in[r_load * N + c_load];
        tile_V[ty + 1][0] = V_in[r_load * N + c_load];
    }
    // Right halo col: tile_U[ty+1][TILE_DIM_X-1], loaded by tx=blockDim.x-1
    if (tx == blockDim.x - 1) {
        r_load = (i_global_raw % N);
        c_load = ((j_global_raw + 1 + N) % N); // Col "right"
        tile_U[ty + 1][TILE_DIM_X - 1] = U_in[r_load * N + c_load];
        tile_V[ty + 1][TILE_DIM_X - 1] = V_in[r_load * N + c_load];
    }

    // 3. Corner halo points, loaded by corner threads.
    // Top-left: tile_U[0][0], by (tx=0, ty=0)
    if (tx == 0 && ty == 0) {
        r_load = ((i_global_raw - 1 + N) % N);
        c_load = ((j_global_raw - 1 + N) % N);
        tile_U[0][0] = U_in[r_load * N + c_load];
        tile_V[0][0] = V_in[r_load * N + c_load];
    }
    // Top-right: tile_U[0][TILE_DIM_X-1], by (tx=blockDim.x-1, ty=0)
    if (tx == blockDim.x - 1 && ty == 0) {
        r_load = ((i_global_raw - 1 + N) % N);
        c_load = ((j_global_raw + 1 + N) % N);
        tile_U[0][TILE_DIM_X - 1] = U_in[r_load * N + c_load];
        tile_V[0][TILE_DIM_X - 1] = V_in[r_load * N + c_load];
    }
    // Bottom-left: tile_U[TILE_DIM_Y-1][0], by (tx=0, ty=blockDim.y-1)
    if (tx == 0 && ty == blockDim.y - 1) {
        r_load = ((i_global_raw + 1 + N) % N);
        c_load = ((j_global_raw - 1 + N) % N);
        tile_U[TILE_DIM_Y - 1][0] = U_in[r_load * N + c_load];
        tile_V[TILE_DIM_Y - 1][0] = V_in[r_load * N + c_load];
    }
    // Bottom-right: tile_U[TILE_DIM_Y-1][TILE_DIM_X-1], by (tx=blockDim.x-1, ty=blockDim.y-1)
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
        r_load = ((i_global_raw + 1 + N) % N);
        c_load = ((j_global_raw + 1 + N) % N);
        tile_U[TILE_DIM_Y - 1][TILE_DIM_X - 1] = U_in[r_load * N + c_load];
        tile_V[TILE_DIM_Y - 1][TILE_DIM_X - 1] = V_in[r_load * N + c_load];
    }

    __syncthreads(); // Synchronize threads after loading shared memory

    // --- Perform computation using shared memory IF thread is within the N x N logical grid ---
    if (i_global_raw < N && j_global_raw < N) {
        // Central point for this thread in shared memory is at (ty+1, tx+1)
        double u_ijk = tile_U[ty + 1][tx + 1];
        double v_ijk = tile_V[ty + 1][tx + 1];

        // Laplacian uses neighbors from shared memory.
        // Accesses are tile_U[ (ty+1) +/- 1 ][ (tx+1) ] and tile_U[ (ty+1) ][ (tx+1) +/- 1 ]
        double laplace_U = tile_U[ty + 1 + 1][tx + 1] + tile_U[ty + 1 - 1][tx + 1] + // tile_U[ty+2][tx+1] (bottom) + tile_U[ty][tx+1] (top)
                           tile_U[ty + 1][tx + 1 + 1] + tile_U[ty + 1][tx + 1 - 1] - // tile_U[ty+1][tx+2] (right) + tile_U[ty+1][tx] (left)
                           4.0 * u_ijk;

        double laplace_V = tile_V[ty + 1 + 1][tx + 1] + tile_V[ty + 1 - 1][tx + 1] +
                           tile_V[ty + 1][tx + 1 + 1] + tile_V[ty + 1][tx + 1 - 1] -
                           4.0 * v_ijk;

        double reaction = u_ijk * v_ijk * v_ijk;

        U_out[i_global_raw * N + j_global_raw] = u_ijk + dt * (Du * laplace_U - reaction + F * (1.0 - u_ijk));
        V_out[i_global_raw * N + j_global_raw] = v_ijk + dt * (Dv * laplace_V + reaction - (F + k) * v_ijk);
    }
}


// --- PGM Saving Function (same as sequential version, adapted for CUDA context) ---
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
            double val = grid[i * N + j]; // Direct 1D indexing
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
    double F = atof(argv[3]);
    double k = atof(argv[4]);

    if (N <= 0 || steps <= 0 || F < 0 || k < 0) { // F and k can be 0
        fprintf(stderr, "Error: N and steps must be positive. F and k must be non-negative.\n");
        return 1;
    }
    printf("INFO: Running CUDA Gray-Scott N=%d, Steps=%d, F=%.4f, k=%.4f\n", N, steps, F, k);

    const double dt = 1.0;
    const double Du = 0.16;
    const double Dv = 0.08;

    size_t gridSize = (size_t)N * N;
    size_t bytes = gridSize * sizeof(double);
    double* h_U_init = (double*)malloc(bytes);
    double* h_V_result = (double*)malloc(bytes);
    if (!h_U_init || !h_V_result) {
        perror("Failed to allocate host memory");
        free(h_U_init); free(h_V_result);
        return 1;
    }
    // printf("INFO: Allocated %.2f MB host memory.\n", 2.0 * bytes / (1024.0 * 1024.0)); // Only if both are kept

    // Initialize Grids on Host
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_U_init[i * N + j] = 1.0;
            h_V_result[i * N + j] = 0.0; // Initialize V here too for d_V copy
        }
    }
    int size = N / 4;
    int start_idx = N / 2 - size / 2;
    int end_idx = start_idx + size;
    if (start_idx < 0) start_idx = 0;
    if (end_idx > N) end_idx = N;

    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = start_idx; j < end_idx; ++j) {
             if (i >= 0 && i < N && j >= 0 && j < N) {
                h_U_init[i * N + j] = 0.75;
                h_V_result[i * N + j] = 0.25;
             }
        }
    }

    double *d_U, *d_V, *d_U_new, *d_V_new;
    CHECK_CUDA(cudaMalloc(&d_U, bytes));
    CHECK_CUDA(cudaMalloc(&d_V, bytes));
    CHECK_CUDA(cudaMalloc(&d_U_new, bytes));
    CHECK_CUDA(cudaMalloc(&d_V_new, bytes));
    // printf("INFO: Allocated %.2f MB device memory.\n", 4.0 * bytes / (1024.0 * 1024.0));

    struct timespec start_time_spec, end_time_spec;
    clock_gettime(CLOCK_MONOTONIC, &start_time_spec);

    CHECK_CUDA(cudaMemcpy(d_U, h_U_init, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V_result, bytes, cudaMemcpyHostToDevice));
    
    free(h_U_init); h_U_init = NULL;

    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // printf("INFO: Launching kernel with gridDim=(%d, %d), blockDim=(%d, %d)\n",
    //       gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    for (int step = 0; step < steps; ++step) {
        gray_scott_kernel<<<gridDim, blockDim>>>(d_U, d_V, d_U_new, d_V_new, N, Du, Dv, F, k, dt);
        CHECK_CUDA(cudaGetLastError()); 

        double* temp_ptr_U = d_U; d_U = d_U_new; d_U_new = temp_ptr_U; // Use distinct temp pointer name
        double* temp_ptr_V = d_V; d_V = d_V_new; d_V_new = temp_ptr_V;
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_V_result, d_V, bytes, cudaMemcpyDeviceToHost)); // Copy final V result to h_V_result

    clock_gettime(CLOCK_MONOTONIC, &end_time_spec);
    double elapsed_time = (end_time_spec.tv_sec - start_time_spec.tv_sec) +
                          (end_time_spec.tv_nsec - start_time_spec.tv_nsec) / 1e9;

    printf("DIMS %dx%d TIME %f\n", N, N, elapsed_time);
    save_pgm(h_V_result, N, steps, F, k, "gray_scott_cuda");

    CHECK_CUDA(cudaFree(d_U));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_U_new));
    CHECK_CUDA(cudaFree(d_V_new));
    free(h_V_result);

    printf("INFO: CUDA simulation finished successfully.\n");
    return 0;
}