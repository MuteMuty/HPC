#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h> // For memcpy, memset
#include <mpi.h>
#include <limits.h> 

// Macro to access 1D array as 2D grid
#define IDX(r, c, N_cols) ((r) * (N_cols) + (c))

// PGM saving function (executed by rank 0) - (keep your existing robust one)
void save_pgm(const double* grid_V_global, int N_global, int steps, double F, double k, const char* prefix) {
    char filename[256];
    snprintf(filename, sizeof(filename), "%s_N%d_steps%d_F%.3f_k%.3f_V.pgm", prefix, N_global, steps, F, k);

    FILE* f = fopen(filename, "w");
    if (!f) {
        perror("Failed to open output PGM file");
        fprintf(stderr, "Filename attempted: %s\n", filename);
        return;
    }

    fprintf(f, "P2\n%d %d\n255\n", N_global, N_global);

    double min_val_actual = grid_V_global[0], max_val_actual = grid_V_global[0];
    if (N_global > 0) { // Ensure grid is not empty
        for (int i = 0; i < N_global * N_global; ++i) {
            if (grid_V_global[i] < min_val_actual) min_val_actual = grid_V_global[i];
            if (grid_V_global[i] > max_val_actual) max_val_actual = grid_V_global[i];
        }
    } else { // Handle empty grid case if it somehow occurs
        min_val_actual = 0.0; max_val_actual = 1.0;
    }
    
    double min_val_norm = min_val_actual;
    double max_val_norm = max_val_actual;

    if (max_val_norm - min_val_norm < 1e-9) { 
        if (fabs(min_val_norm) < 1e-9) { 
             min_val_norm = 0.0; 
             max_val_norm = 1.0; 
        } else {
            max_val_norm = min_val_norm + 1e-6; 
        }
    }
    
    int count = 0;
    for (int i = 0; i < N_global; ++i) {
        for (int j = 0; j < N_global; ++j) {
            double val = grid_V_global[IDX(i, j, N_global)];
            double normalized_v = (max_val_norm - min_val_norm == 0) ? 0 : (val - min_val_norm) / (max_val_norm - min_val_norm);
            int gray_val = (int)(255.0 * normalized_v);
            
            if (gray_val < 0) gray_val = 0;
            if (gray_val > 255) gray_val = 255;

            fprintf(f, "%d", gray_val);
            count++;
            if (count % 16 == 0 || j == N_global - 1) {
                fprintf(f, "\n");
            } else {
                fprintf(f, " ");
            }
        }
    }
    fclose(f);
}


// Calculates the number of rows this rank is responsible for and its global starting row index
void calculate_local_rows_and_start(int N_global, int rank, int size, int* local_N_rows, int* start_row_global) {
    int base_rows = N_global / size;
    int remainder_rows = N_global % size;
    *local_N_rows = base_rows + (rank < remainder_rows ? 1 : 0);
    *start_row_global = rank * base_rows + (rank < remainder_rows ? rank : remainder_rows);
}

// Initializes the local portions of U and V grids for the current rank
void initialize_grids_local(double* U_local, double* V_local, int local_N_rows, int N_cols,
                            int start_row_global, int N_global_total) {
    for (int r_local_with_ghost = 0; r_local_with_ghost < local_N_rows + 2; ++r_local_with_ghost) {
        for (int c = 0; c < N_cols; ++c) {
            U_local[IDX(r_local_with_ghost, c, N_cols)] = 1.0;
            V_local[IDX(r_local_with_ghost, c, N_cols)] = 0.0;
        }
    }

    int patch_size = N_global_total / 4;
    int patch_start_global_row = N_global_total / 2 - patch_size / 2;
    int patch_end_global_row = patch_start_global_row + patch_size;
    int patch_start_global_col = N_global_total / 2 - patch_size / 2;
    int patch_end_global_col = patch_start_global_col + patch_size;

    for (int r_local_idx = 0; r_local_idx < local_N_rows; ++r_local_idx) {
        int r_global = start_row_global + r_local_idx; 
        if (r_global >= patch_start_global_row && r_global < patch_end_global_row) {
            for (int c = 0; c < N_cols; ++c) {
                if (c >= patch_start_global_col && c < patch_end_global_col) {
                    U_local[IDX(r_local_idx + 1, c, N_cols)] = 0.75;
                    V_local[IDX(r_local_idx + 1, c, N_cols)] = 0.25;
                }
            }
        }
    }
}

// NEW exchange_halo_rows using Isend/Irecv
void exchange_halo_rows_isend_irecv(double* grid_local, int local_N_rows, int N_cols, int rank, int size, MPI_Comm comm) {
    if (size == 1) {
        memcpy(&grid_local[IDX(0, 0, N_cols)], &grid_local[IDX(local_N_rows, 0, N_cols)], N_cols * sizeof(double));
        memcpy(&grid_local[IDX(local_N_rows + 1, 0, N_cols)], &grid_local[IDX(1, 0, N_cols)], N_cols * sizeof(double));
        return;
    }

    int proc_up = (rank - 1 + size) % size;
    int proc_down = (rank + 1) % size;
    MPI_Request reqs[4];
    MPI_Status stats[4]; // Can be MPI_STATUSES_IGNORE if not checking status fields

    const int tag_going_up = 0;   // Data sent from rank r to rank r-1 (proc_up)
    const int tag_going_down = 1; // Data sent from rank r to rank r+1 (proc_down)

    // Send my actual top row (grid_local[1]) UP to proc_up
    MPI_Isend(&grid_local[IDX(1, 0, N_cols)], N_cols, MPI_DOUBLE, proc_up, tag_going_up, comm, &reqs[0]);
    
    // Send my actual bottom row (grid_local[local_N_rows]) DOWN to proc_down
    MPI_Isend(&grid_local[IDX(local_N_rows, 0, N_cols)], N_cols, MPI_DOUBLE, proc_down, tag_going_down, comm, &reqs[1]);

    // Receive into my top ghost row (grid_local[0]) from proc_up. 
    // proc_up sent this data DOWN to me (its proc_down relative to itself), so it used tag_going_down for its send.
    MPI_Irecv(&grid_local[IDX(0, 0, N_cols)], N_cols, MPI_DOUBLE, proc_up, tag_going_down, comm, &reqs[2]);
    
    // Receive into my bottom ghost row (grid_local[local_N_rows+1]) from proc_down.
    // proc_down sent this data UP to me (its proc_up relative to itself), so it used tag_going_up for its send.
    MPI_Irecv(&grid_local[IDX(local_N_rows + 1, 0, N_cols)], N_cols, MPI_DOUBLE, proc_down, tag_going_up, comm, &reqs[3]);

    MPI_Waitall(4, reqs, stats); // or MPI_STATUSES_IGNORE for stats
}


// Computes one step of the Gray-Scott simulation for the local grid portion
void compute_gray_scott_local_step(
    const double* U_curr_local, const double* V_curr_local,
    double* U_next_local, double* V_next_local,
    int local_N_rows, int N_cols,
    double Du, double Dv, double F, double k, double dt) {

    for (int r_local = 1; r_local <= local_N_rows; ++r_local) {
        for (int c = 0; c < N_cols; ++c) {
            double u_ijk = U_curr_local[IDX(r_local, c, N_cols)];
            double v_ijk = V_curr_local[IDX(r_local, c, N_cols)];

            double laplace_U = U_curr_local[IDX(r_local - 1, c, N_cols)] +
                               U_curr_local[IDX(r_local + 1, c, N_cols)] +
                               U_curr_local[IDX(r_local, (c - 1 + N_cols) % N_cols, N_cols)] + 
                               U_curr_local[IDX(r_local, (c + 1) % N_cols, N_cols)] -
                               4.0 * u_ijk;

            double laplace_V = V_curr_local[IDX(r_local - 1, c, N_cols)] +
                               V_curr_local[IDX(r_local + 1, c, N_cols)] +
                               V_curr_local[IDX(r_local, (c - 1 + N_cols) % N_cols, N_cols)] +
                               V_curr_local[IDX(r_local, (c + 1) % N_cols, N_cols)] -
                               4.0 * v_ijk;
            
            double reaction = u_ijk * v_ijk * v_ijk;

            U_next_local[IDX(r_local, c, N_cols)] = u_ijk + dt * (Du * laplace_U - reaction + F * (1.0 - u_ijk));
            V_next_local[IDX(r_local, c, N_cols)] = v_ijk + dt * (Dv * laplace_V + reaction - (F + k) * v_ijk);
        }
    }
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 5) {
        if (world_rank == 0) {
            fprintf(stderr, "Usage: %s <N_global> <steps> <F> <k>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int N_global = atoi(argv[1]);
    int steps = atoi(argv[2]);
    double F_param = atof(argv[3]);
    double k_param = atof(argv[4]);

    if (N_global <= 0 || steps <= 0 || F_param < 0 || k_param < 0) {
        if (world_rank == 0) {
            fprintf(stderr, "Error: N_global and steps must be positive. F and k must be non-negative.\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    if (world_rank == 0) {
        printf("INFO: MPI Gray-Scott N=%d, Steps=%d, F=%.4f, k=%.4f, Processes=%d (Using Isend/Irecv)\n",
               N_global, steps, F_param, k_param, world_size);
    }

    const double dt = 1.0;
    const double Du = 0.16;
    const double Dv = 0.08;

    int local_N_rows, start_row_global;
    calculate_local_rows_and_start(N_global, world_rank, world_size, &local_N_rows, &start_row_global);
    
    size_t local_grid_size_with_ghost = (size_t)(local_N_rows + 2) * N_global;
    double* U_curr_local = NULL;
    double* V_curr_local = NULL;
    double* U_next_local = NULL;
    double* V_next_local = NULL;

    if (local_N_rows > 0 || world_size == 1) {
        U_curr_local = (double*)malloc(local_grid_size_with_ghost * sizeof(double));
        V_curr_local = (double*)malloc(local_grid_size_with_ghost * sizeof(double));
        U_next_local = (double*)malloc(local_grid_size_with_ghost * sizeof(double));
        V_next_local = (double*)malloc(local_grid_size_with_ghost * sizeof(double));

        if (!U_curr_local || !V_curr_local || !U_next_local || !V_next_local) {
            perror("Failed to allocate local grid memory"); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        initialize_grids_local(U_curr_local, V_curr_local, local_N_rows, N_global, start_row_global, N_global);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_wtime = 0, end_wtime = 0;
    if (world_rank == 0) {
        start_wtime = MPI_Wtime();
    }

    for (int s = 0; s < steps; ++s) {
        // if (world_rank == 0 && s % 500 == 0) {printf("Rank 0 on step %d\n",s); fflush(stdout);}

        if (local_N_rows > 0 || world_size == 1) { 
            exchange_halo_rows_isend_irecv(U_curr_local, local_N_rows, N_global, world_rank, world_size, MPI_COMM_WORLD);
            exchange_halo_rows_isend_irecv(V_curr_local, local_N_rows, N_global, world_rank, world_size, MPI_COMM_WORLD);

            compute_gray_scott_local_step(U_curr_local, V_curr_local, U_next_local, V_next_local,
                                          local_N_rows, N_global, Du, Dv, F_param, k_param, dt);
            
            double* temp_U = U_curr_local; U_curr_local = U_next_local; U_next_local = temp_U;
            double* temp_V = V_curr_local; V_curr_local = V_next_local; V_next_local = temp_V;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); 

    double* V_global = NULL;
    int* recvcounts = NULL;
    int* displs = NULL;

    if (world_rank == 0) {
        V_global = (double*)malloc((size_t)N_global * N_global * sizeof(double));
        if (!V_global) { perror("Failed to allocate V_global"); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); }
        
        recvcounts = (int*)malloc(world_size * sizeof(int));
        displs = (int*)malloc(world_size * sizeof(int));
        if (!recvcounts || !displs) { perror("Failed to allocate recvcounts/displs"); MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); }

        int current_displ = 0;
        for (int i = 0; i < world_size; ++i) {
            int proc_local_N_rows, proc_start_row; 
            calculate_local_rows_and_start(N_global, i, world_size, &proc_local_N_rows, &proc_start_row);
            recvcounts[i] = proc_local_N_rows * N_global; 
            displs[i] = current_displ;
            current_displ += recvcounts[i];
        }
    }
    
    double* send_buffer_for_gather = NULL;
    int send_count_for_gather = 0;
    if (local_N_rows > 0 || world_size == 1) { 
        send_buffer_for_gather = &V_curr_local[IDX(1, 0, N_global)];
        send_count_for_gather = local_N_rows * N_global;
    }
    
    MPI_Gatherv(send_buffer_for_gather, send_count_for_gather, MPI_DOUBLE,
                V_global, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        end_wtime = MPI_Wtime(); 
        printf("DIMS %dx%d CORES %d TIME %f\n", N_global, N_global, world_size, end_wtime - start_wtime);
        
        char pgm_filename_buf[100]; 
        snprintf(pgm_filename_buf, sizeof(pgm_filename_buf), "gray_scott_mpi_N%d_steps%d_F%.3f_k%.3f_V.pgm", N_global, steps, F_param, k_param);
        if (N_global > 0) { // Only save if grid is not empty
            save_pgm(V_global, N_global, steps, F_param, k_param, "gray_scott_mpi");
            printf("INFO: Rank 0 saved V grid to %s\n", pgm_filename_buf);
        }
        
        free(V_global);
        free(recvcounts);
        free(displs);
    }

    if (local_N_rows > 0 || world_size == 1) {
        free(U_curr_local); 
        free(V_curr_local); 
        free(U_next_local); 
        free(V_next_local); 
    }

    MPI_Finalize();
    return 0;
}