#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> // For memset

// Macro to access grid elements (handles 1D array indexing)
#define IDX(i, j, N) ((i) * (N) + (j))

// Function to save the V grid as a PGM image
void save_pgm(const double* grid, int N, const char* filename) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        perror("Failed to open output file");
        return;
    }

    // PGM header: P2 format (grayscale text), width, height, max value
    fprintf(f, "P2\n%d %d\n255\n", N, N);

    double min_val = grid[0], max_val = grid[0];
    for (int i = 0; i < N * N; ++i) {
        if (grid[i] < min_val) min_val = grid[i];
        if (grid[i] > max_val) max_val = grid[i];
    }
     // Handle case where all values are the same
    if (max_val <= min_val) max_val = min_val + 1e-6;


    int count = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            // Scale V concentration (typically 0 to ~1) to 0-255 grayscale
            double val = grid[IDX(i, j, N)];
            int gray_val = (int)(255.0 * (val - min_val) / (max_val - min_val));
            // Clamp value just in case
            if (gray_val < 0) gray_val = 0;
            if (gray_val > 255) gray_val = 255;

            fprintf(f, "%d", gray_val);
            count++;
            if (count % 16 == 0 || j == N - 1) { // Newline every 16 values or at end of row
                 fprintf(f, "\n");
            } else {
                 fprintf(f, " ");
            }
        }
    }

    fclose(f);
    printf("INFO: Saved V grid to %s\n", filename);
}


// Function to perform the Gray-Scott simulation
void gray_scott_solver(double* U, double* V, double* U_new, double* V_new,
                       int N, double Du, double Dv, double F, double k, double dt, int steps)
{
    for (int step = 0; step < steps; ++step) {
        // Loop over each cell in the grid
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                // Get current U and V values
                double u_ijk = U[IDX(i, j, N)];
                double v_ijk = V[IDX(i, j, N)];

                // Calculate Laplacian using 5-point stencil with periodic boundary conditions
                // Indices with wrapping: (index + N) % N handles negative results correctly
                int i_plus_1 = (i + 1) % N;
                int i_minus_1 = (i - 1 + N) % N;
                int j_plus_1 = (j + 1) % N;
                int j_minus_1 = (j - 1 + N) % N;

                double laplace_U = U[IDX(i_plus_1, j, N)] + U[IDX(i_minus_1, j, N)] +
                                   U[IDX(i, j_plus_1, N)] + U[IDX(i, j_minus_1, N)] -
                                   4.0 * u_ijk;

                double laplace_V = V[IDX(i_plus_1, j, N)] + V[IDX(i_minus_1, j, N)] +
                                   V[IDX(i, j_plus_1, N)] + V[IDX(i, j_minus_1, N)] -
                                   4.0 * v_ijk;

                // Calculate reaction term
                double reaction = u_ijk * v_ijk * v_ijk;

                // Apply Gray-Scott update rules (Forward Euler)
                U_new[IDX(i, j, N)] = u_ijk + dt * (Du * laplace_U - reaction + F * (1.0 - u_ijk));
                V_new[IDX(i, j, N)] = v_ijk + dt * (Dv * laplace_V + reaction - (F + k) * v_ijk);

                // Optional: Clamp values to prevent numerical instability (especially with larger dt)
                // if (U_new[IDX(i,j,N)] < 0.0) U_new[IDX(i,j,N)] = 0.0;
                // if (U_new[IDX(i,j,N)] > 1.0) U_new[IDX(i,j,N)] = 1.0;
                // if (V_new[IDX(i,j,N)] < 0.0) V_new[IDX(i,j,N)] = 0.0;
                // if (V_new[IDX(i,j,N)] > 1.0) V_new[IDX(i,j,N)] = 1.0;
            }
        }

        // Swap grids for the next iteration
        double* temp_U = U;
        U = U_new;
        U_new = temp_U;

        double* temp_V = V;
        V = V_new;
        V_new = temp_V;

        // Optional: Print progress
        // if ((step + 1) % 100 == 0) {
        //     printf("Step %d completed\n", step + 1);
        // }
    }
     // Important: Ensure the final result is in the original U and V pointers
     // If 'steps' is odd, the final result is in U_new/V_new. We need to copy it back
     // or adjust the pointers passed back to main if that was the design.
     // Since we swapped pointers *inside* this function, the pointers U and V
     // now point to the buffers containing the final result if 'steps' is even.
     // If 'steps' is odd, U_new and V_new point to the final result.
     // The easiest way is to ensure the original pointers passed from main always
     // hold the final result after the function returns.
     if (steps % 2 != 0) {
         // If steps is odd, the result is in the 'new' buffers. Copy it back.
         // (Alternatively, we could swap pointers back one last time, but this is safer
         // if the caller expects the result in the originally named buffers).
         // Note: U_new and V_new pointers here actually point to the *original* U/V buffers passed from main
         // because of the last swap inside the loop when step = steps-1.
         // And U, V point to the 'new' buffers allocated in main.
         // So, copy from the buffers currently pointed to by U/V into the buffers
         // pointed to by U_new/V_new (which are the original U/V buffers).
         memcpy(U_new, U, N * N * sizeof(double));
         memcpy(V_new, V, N * N * sizeof(double));
     }
}


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

    // Simulation parameters
    const double dt = 1.0;
    const double Du = 0.16;
    const double Dv = 0.08;
    const double F = 0.060; // Feed rate
    const double k = 0.062; // Kill rate

    // Allocate memory for the grids
    size_t gridSize = (size_t)N * N;
    double* U = (double*)malloc(gridSize * sizeof(double));
    double* V = (double*)malloc(gridSize * sizeof(double));
    double* U_new = (double*)malloc(gridSize * sizeof(double));
    double* V_new = (double*)malloc(gridSize * sizeof(double));

    if (!U || !V || !U_new || !V_new) {
        perror("Failed to allocate memory for grids");
        free(U); free(V); free(U_new); free(V_new);
        return 1;
    }

    // Initialize grids
    // U = 1.0 everywhere, V = 0.0 everywhere initially
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            U[IDX(i, j, N)] = 1.0;
            V[IDX(i, j, N)] = 0.0;
        }
    }

    // Seed the center N/4 x N/4 square
    int size = N / 4;
    int start = N / 2 - size / 2;
    int end = N / 2 + size / 2; // End index is exclusive or inclusive depending on interpretation, let's make it inclusive for simplicity
     if (start < 0) start = 0;
     if (end > N) end = N;


    for (int i = start; i < end; ++i) {
        for (int j = start; j < end; ++j) {
             // Check bounds just in case N/4 is odd leading to off-by-one
             if (i >= 0 && i < N && j >= 0 && j < N) {
                U[IDX(i, j, N)] = 0.75; // Assignment says 0.75 but examples often use 0.5
                V[IDX(i, j, N)] = 0.25;
             }
        }
    }


    // --- Time Measurement Start ---
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Run the simulation
    // Pass the pointers to the initially allocated buffers
    gray_scott_solver(U, V, U_new, V_new, N, Du, Dv, F, k, dt, steps);

    // --- Time Measurement End ---
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    double elapsed_time = (end_time.tv_sec - start_time.tv_sec) +
                          (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    // Print results in the required format for the script
    // DIMS NxN TIME time_in_seconds
    printf("DIMS %dx%d TIME %f\n", N, N, elapsed_time);

    // Save the final V grid state
    char filename[100];
    snprintf(filename, sizeof(filename), "sequential_N%d_steps%d_V.pgm", N, steps);
    // The final result is guaranteed to be in the original U and V buffers
    // due to the handling within gray_scott_solver
    save_pgm(V, N, filename);


    // Free memory
    free(U);
    free(V);
    free(U_new);
    free(V_new);

    return 0;
}