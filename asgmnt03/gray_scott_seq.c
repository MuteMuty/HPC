#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> // For memset, memcpy

// Macro to access grid elements (handles 1D array indexing)
#define IDX(i, j, N) ((i) * (N) + (j))

// Function to save the V grid as a PGM image
void save_pgm(const double* grid, int N, int steps, double F, double k, const char* prefix) {
    char filename[150];
    snprintf(filename, sizeof(filename), "%s_N%d_steps%d_F%.3f_k%.3f_V.pgm", prefix, N, steps, F, k);

    FILE* f = fopen(filename, "w");
    if (!f) {
        perror("Failed to open output file");
        fprintf(stderr, "Filename attempted: %s\n", filename);
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


void gray_scott_solver(double* U, double* V, double* U_new, double* V_new,
                       int N, double Du, double Dv, double F, double k, double dt, int steps)
{
    for (int step = 0; step < steps; ++step) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double u_ijk = U[IDX(i, j, N)];
                double v_ijk = V[IDX(i, j, N)];

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

                double reaction = u_ijk * v_ijk * v_ijk;

                U_new[IDX(i, j, N)] = u_ijk + dt * (Du * laplace_U - reaction + F * (1.0 - u_ijk));
                V_new[IDX(i, j, N)] = v_ijk + dt * (Dv * laplace_V + reaction - (F + k) * v_ijk);
            }
        }

        double* temp_U = U; U = U_new; U_new = temp_U;
        double* temp_V = V; V = V_new; V_new = temp_V;
    }
    // Ensure final result is in original U/V buffers passed from main
    if (steps % 2 != 0) {
        memcpy(U_new, U, (size_t)N * N * sizeof(double));
        memcpy(V_new, V, (size_t)N * N * sizeof(double));
    }
}


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
    double F = atof(argv[3]); // Feed rate
    double k = atof(argv[4]); // Kill rate

    if (N <= 0 || steps <= 0 || F < 0 || k < 0) {
        fprintf(stderr, "Error: N and steps must be positive. F and k must be non-negative.\n");
        return 1;
    }
    printf("INFO: Running Sequential Gray-Scott N=%d, Steps=%d, F=%.4f, k=%.4f\n", N, steps, F, k);


    const double dt = 1.0;
    const double Du = 0.16;
    const double Dv = 0.08;

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

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            U[IDX(i, j, N)] = 1.0;
            V[IDX(i, j, N)] = 0.0;
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
                U[IDX(i, j, N)] = 0.75; // The example text used 0.5, assignment text 0.75.
                V[IDX(i, j, N)] = 0.25;
             }
        }
    }


    // --- Time Measurement Start ---
    struct timespec start_time, end_time_spec;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    gray_scott_solver(U, V, U_new, V_new, N, Du, Dv, F, k, dt, steps);

    // --- Time Measurement End ---
    clock_gettime(CLOCK_MONOTONIC, &end_time_spec);

    double elapsed_time = (end_time_spec.tv_sec - start_time.tv_sec) +
                          (end_time_spec.tv_nsec - start_time.tv_nsec) / 1e9;

    printf("DIMS %dx%d TIME %f\n", N, N, elapsed_time);

    save_pgm(V, N, steps, F, k, "gray_scott_seq");

    free(U);
    free(V);
    free(U_new);
    free(V_new);

    return 0;
}