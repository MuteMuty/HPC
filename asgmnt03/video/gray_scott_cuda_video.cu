#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>      // For memcpy
#include <vector>        // For std::vector (used in save_ppm min/max finding)
#include <algorithm>     // For std::minmax_element
#include <limits>        // For numeric_limits
#include <sys/stat.h>    // For mkdir
#include <sys/types.h>   // For mkdir
#include <cuda_runtime.h>

// --- CUDA Error Checking Macro ---
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// --- Shared Memory Tile Dimensions (Keep from previous version) ---
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define TILE_DIM_X (BLOCK_SIZE_X + 2)
#define TILE_DIM_Y (BLOCK_SIZE_Y + 2)

// --- Gray-Scott Kernel (Keep corrected version from previous response) ---
__global__ void gray_scott_kernel(const double* __restrict__ U_in,
                                  const double* __restrict__ V_in,
                                  double* __restrict__ U_out,
                                  double* __restrict__ V_out,
                                  int N, double Du, double Dv, double F, double k, double dt)
{
    __shared__ double tile_U[TILE_DIM_Y][TILE_DIM_X];
    __shared__ double tile_V[TILE_DIM_Y][TILE_DIM_X];

    const int j_global_raw = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_global_raw = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // --- Load data into shared memory with periodic wrapping ---
    // (Use the corrected loading logic from the previous response)
    int r_load = (i_global_raw % N);
    int c_load = (j_global_raw % N);
    tile_U[ty + 1][tx + 1] = U_in[r_load * N + c_load];
    tile_V[ty + 1][tx + 1] = V_in[r_load * N + c_load];
    if (ty == 0) {
        r_load = ((i_global_raw - 1 + N) % N); c_load = (j_global_raw % N);
        tile_U[0][tx + 1] = U_in[r_load * N + c_load];
        tile_V[0][tx + 1] = V_in[r_load * N + c_load];
    }
    if (ty == blockDim.y - 1) {
        r_load = ((i_global_raw + 1 + N) % N); c_load = (j_global_raw % N);
        tile_U[TILE_DIM_Y - 1][tx + 1] = U_in[r_load * N + c_load];
        tile_V[TILE_DIM_Y - 1][tx + 1] = V_in[r_load * N + c_load];
    }
    if (tx == 0) {
        r_load = (i_global_raw % N); c_load = ((j_global_raw - 1 + N) % N);
        tile_U[ty + 1][0] = U_in[r_load * N + c_load];
        tile_V[ty + 1][0] = V_in[r_load * N + c_load];
    }
    if (tx == blockDim.x - 1) {
        r_load = (i_global_raw % N); c_load = ((j_global_raw + 1 + N) % N);
        tile_U[ty + 1][TILE_DIM_X - 1] = U_in[r_load * N + c_load];
        tile_V[ty + 1][TILE_DIM_X - 1] = V_in[r_load * N + c_load];
    }
    if (tx == 0 && ty == 0) {
        r_load = ((i_global_raw - 1 + N) % N); c_load = ((j_global_raw - 1 + N) % N);
        tile_U[0][0] = U_in[r_load * N + c_load];
        tile_V[0][0] = V_in[r_load * N + c_load];
    }
    if (tx == blockDim.x - 1 && ty == 0) {
        r_load = ((i_global_raw - 1 + N) % N); c_load = ((j_global_raw + 1 + N) % N);
        tile_U[0][TILE_DIM_X - 1] = U_in[r_load * N + c_load];
        tile_V[0][TILE_DIM_X - 1] = V_in[r_load * N + c_load];
    }
    if (tx == 0 && ty == blockDim.y - 1) {
        r_load = ((i_global_raw + 1 + N) % N); c_load = ((j_global_raw - 1 + N) % N);
        tile_U[TILE_DIM_Y - 1][0] = U_in[r_load * N + c_load];
        tile_V[TILE_DIM_Y - 1][0] = V_in[r_load * N + c_load];
    }
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
        r_load = ((i_global_raw + 1 + N) % N); c_load = ((j_global_raw + 1 + N) % N);
        tile_U[TILE_DIM_Y - 1][TILE_DIM_X - 1] = U_in[r_load * N + c_load];
        tile_V[TILE_DIM_Y - 1][TILE_DIM_X - 1] = V_in[r_load * N + c_load];
    }
    __syncthreads();

    // --- Perform computation using shared memory IF thread is within the N x N logical grid ---
    if (i_global_raw < N && j_global_raw < N) {
        double u_ijk = tile_U[ty + 1][tx + 1];
        double v_ijk = tile_V[ty + 1][tx + 1];
        double laplace_U = tile_U[ty + 2][tx + 1] + tile_U[ty][tx + 1] +
                           tile_U[ty + 1][tx + 2] + tile_U[ty + 1][tx] - 4.0 * u_ijk;
        double laplace_V = tile_V[ty + 2][tx + 1] + tile_V[ty][tx + 1] +
                           tile_V[ty + 1][tx + 2] + tile_V[ty + 1][tx] - 4.0 * v_ijk;
        double reaction = u_ijk * v_ijk * v_ijk;
        U_out[i_global_raw * N + j_global_raw] = u_ijk + dt * (Du * laplace_U - reaction + F * (1.0 - u_ijk));
        V_out[i_global_raw * N + j_global_raw] = v_ijk + dt * (Dv * laplace_V + reaction - (F + k) * v_ijk);
    }
}


// --- Simple Colormap Function (Blue -> Yellow -> Red) ---
void simple_colormap(double v, unsigned char& r, unsigned char& g, unsigned char& b) {
    v = fmax(0.0, fmin(1.0, v)); // Clamp v to [0, 1]
    if (v < 0.5) {
        double t = v * 2.0;
        r = (unsigned char)(0 * (1.0 - t) + 255 * t);
        g = (unsigned char)(0 * (1.0 - t) + 255 * t);
        b = (unsigned char)(255 * (1.0 - t) + 0 * t);
    } else {
        double t = (v - 0.5) * 2.0;
        r = 255;
        g = (unsigned char)(255 * (1.0 - t) + 0 * t);
        b = 0;
    }
}

// --- PPM Saving Function (Color) ---
void save_ppm(const double* grid_v, int N, int frame_num, const char* frame_dir) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%s/frame_%05d.ppm", frame_dir, frame_num);

    FILE* f = fopen(filename, "wb"); // Use "wb" for potentially faster binary write? (Text mode should be fine too)
    if (!f) {
        perror("Failed to open output PPM file");
        fprintf(stderr, "Filename attempted: %s\n", filename);
        return;
    }

    fprintf(f, "P3\n%d %d\n255\n", N, N);

    // Find min/max V values for normalization
    std::vector<double> v_vec(grid_v, grid_v + (size_t)N*N);
    auto minmax = std::minmax_element(v_vec.begin(), v_vec.end());
    double min_val = *minmax.first;
    double max_val = *minmax.second;
    double range = max_val - min_val;
    if (range <= std::numeric_limits<double>::epsilon()) {
        range = 1.0;
        min_val = 0.0;
    }

    // Prepare buffer for one row of PPM data (optional optimization)
    // std::vector<char> row_buffer(N * 3 * 4); // Approx size needed

    for (int i = 0; i < N; ++i) {
        // row_buffer.clear(); // Clear buffer for new row
        for (int j = 0; j < N; ++j) {
            double val = grid_v[i * N + j];
            double normalized_v = (val - min_val) / range;

            unsigned char r, g, b;
            simple_colormap(normalized_v, r, g, b);

            fprintf(f, "%d %d %d ", r, g, b);
            // Optional: Buffer row data
            // char triplet[15];
            // snprintf(triplet, sizeof(triplet), "%d %d %d ", r, g, b);
            // row_buffer.insert(row_buffer.end(), triplet, triplet + strlen(triplet));
        }
        fprintf(f, "\n");
        // Optional: Write buffered row
        // row_buffer.push_back('\n');
        // fwrite(row_buffer.data(), sizeof(char), row_buffer.size(), f);
    }

    fclose(f);
}

// --- Main Function ---
int main(int argc, char* argv[]) {
    // N steps F k pattern_name frame_interval
    if (argc != 7) {
        fprintf(stderr, "Usage: %s <N> <steps> <F> <k> <pattern_name> <frame_interval>\n", argv[0]);
        fprintf(stderr, "  N: grid size (intended for 256)\n");
        fprintf(stderr, "  steps: number of simulation steps\n");
        fprintf(stderr, "  F: feed rate\n");
        fprintf(stderr, "  k: kill rate\n");
        fprintf(stderr, "  pattern_name: Name for output directory (e.g., 'Default')\n");
        fprintf(stderr, "  frame_interval: Save frame every N steps\n");
        return 1;
    }

    int N = atoi(argv[1]);
    int steps = atoi(argv[2]);
    double F = atof(argv[3]);
    double k = atof(argv[4]);
    const char* pattern_name = argv[5];
    int frame_interval = atoi(argv[6]);

    if (N <= 0 || steps <= 0 || F < 0 || k < 0 || frame_interval <= 0) {
        fprintf(stderr, "Error: N, steps, frame_interval must be positive. F, k must be non-negative.\n");
        return 1;
    }
     if (N != 256) {
        fprintf(stderr, "Warning: This executable is optimized/intended for N=256. Running with N=%d.\n", N);
    }

    printf("INFO: Running CUDA Gray-Scott Video Frame Generation\n");
    printf("      N=%d, Steps=%d, F=%.4f, k=%.4f, Pattern=%s, Interval=%d\n",
           N, steps, F, k, pattern_name, frame_interval);

    // --- Simulation Parameters ---
    const double dt = 1.0;
    const double Du = 0.16;
    const double Dv = 0.08;

    // --- Create Frame Directory ---
    char frame_dir[200];
    snprintf(frame_dir, sizeof(frame_dir), "frames_%s_N%d_F%.3f_k%.3f", pattern_name, N, F, k);
    if (mkdir(frame_dir, 0777) == -1) {
        // Ignore error if directory already exists, fail otherwise
        /* if (errno != EEXIST) {
            perror("Failed to create frame directory");
            fprintf(stderr, "Directory attempted: %s\n", frame_dir);
            return 1;
        } */
        printf("INFO: Frame directory '%s' already exists. Overwriting frames.\n", frame_dir);
    } else {
        printf("INFO: Created frame directory: %s\n", frame_dir);
    }


    // --- Memory Allocation ---
    size_t gridSize = (size_t)N * N;
    size_t bytes = gridSize * sizeof(double);
    double* h_U_init = (double*)malloc(bytes);
    double* h_V_buffer = (double*)malloc(bytes); // Buffer for init V and frames
    if (!h_U_init || !h_V_buffer) {
        perror("Failed to allocate host memory");
        free(h_U_init); free(h_V_buffer); return 1;
    }

    double *d_U, *d_V, *d_U_new, *d_V_new;
    CHECK_CUDA(cudaMalloc(&d_U, bytes));
    CHECK_CUDA(cudaMalloc(&d_V, bytes));
    CHECK_CUDA(cudaMalloc(&d_U_new, bytes));
    CHECK_CUDA(cudaMalloc(&d_V_new, bytes));

    // --- Initialize Grids on Host ---
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_U_init[i * N + j] = 1.0;
            h_V_buffer[i * N + j] = 0.0;
        }
    }
    int size = N / 4;
    int start_idx = N / 2 - size / 2;
    int end_idx = start_idx + size;
     if (start_idx < 0) start_idx = 0; if (end_idx > N) end_idx = N;
    for (int i = start_idx; i < end_idx; ++i) {
        for (int j = start_idx; j < end_idx; ++j) {
             if (i >= 0 && i < N && j >= 0 && j < N) {
                h_U_init[i * N + j] = 0.75;
                h_V_buffer[i * N + j] = 0.25;
             }
        }
    }

    // --- Copy Initial Data H->D ---
    CHECK_CUDA(cudaMemcpy(d_U, h_U_init, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V_buffer, bytes, cudaMemcpyHostToDevice));
    free(h_U_init); h_U_init = NULL;

    // --- Kernel Launch Config ---
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    printf("INFO: Starting simulation loop...\n");
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    int frame_count = 0;

    // --- Simulation Loop with Frame Saving ---
    for (int step = 0; step < steps; ++step) {

        // Save frame: Initial state (step 0) and at intervals
         if (step % frame_interval == 0) {
             printf("INFO: Saving frame %d (step %d)...\n", frame_count, step);
             CHECK_CUDA(cudaMemcpy(h_V_buffer, d_V, bytes, cudaMemcpyDeviceToHost));
             // CHECK_CUDA(cudaDeviceSynchronize()); // Sync *before* saving might be safer? Optional.
             save_ppm(h_V_buffer, N, frame_count, frame_dir);
             frame_count++;
         }

        // Compute next step
        gray_scott_kernel<<<gridDim, blockDim>>>(d_U, d_V, d_U_new, d_V_new, N, Du, Dv, F, k, dt);
        CHECK_CUDA(cudaGetLastError());

        // Swap pointers
        double* temp_ptr_U = d_U; d_U = d_U_new; d_U_new = temp_ptr_U;
        double* temp_ptr_V = d_V; d_V = d_V_new; d_V_new = temp_ptr_V;
    }

    // Save the final frame if it wasn't saved on the last interval step
    if (steps > 0 && (steps % frame_interval != 0)) {
         printf("INFO: Saving final frame %d (step %d)...\n", frame_count, steps);
         CHECK_CUDA(cudaMemcpy(h_V_buffer, d_V, bytes, cudaMemcpyDeviceToHost));
         // CHECK_CUDA(cudaDeviceSynchronize());
         save_ppm(h_V_buffer, N, frame_count, frame_dir);
         frame_count++;
    }

    CHECK_CUDA(cudaDeviceSynchronize()); // Ensure all GPU work is done

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                          (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    printf("INFO: Simulation loop finished. Total time: %.3f s.\n", elapsed_time);
    printf("INFO: %d frames saved in directory: %s\n", frame_count, frame_dir);

    // --- Cleanup ---
    CHECK_CUDA(cudaFree(d_U));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_U_new));
    CHECK_CUDA(cudaFree(d_V_new));
    free(h_V_buffer);

    printf("INFO: CUDA frame generation finished successfully.\n");
    return 0;
}