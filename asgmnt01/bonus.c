#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 3

// Function to compute energy using color difference gradient
void compute_energy(const unsigned char *image, int width, int height, int *energy) {
    #pragma omp parallel for collapse(2) schedule(guided)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const unsigned char *pixel = image + (y * width + x) * COLOR_CHANNELS;
            
            // Calculate horizontal gradient
            int left_x = (x > 0) ? x - 1 : x;
            int right_x = (x < width - 1) ? x + 1 : x;
            const unsigned char *left = image + (y * width + left_x) * COLOR_CHANNELS;
            const unsigned char *right = image + (y * width + right_x) * COLOR_CHANNELS;
            int dx = abs(left[0] - right[0]) + abs(left[1] - right[1]) + abs(left[2] - right[2]);

            // Calculate vertical gradient
            int top_y = (y > 0) ? y - 1 : y;
            int bottom_y = (y < height - 1) ? y + 1 : y;
            const unsigned char *top = image + (top_y * width + x) * COLOR_CHANNELS;
            const unsigned char *bottom = image + (bottom_y * width + x) * COLOR_CHANNELS;
            int dy = abs(top[0] - bottom[0]) + abs(top[1] - bottom[1]) + abs(top[2] - bottom[2]);

            energy[y * width + x] = dx + dy;
        }
    }
}

// Function to find seams using dynamic programming with triangular-blocked approach
void find_seam_triangular(const int *energy, int width, int height, int *seam) {
    int *dp = (int *)malloc(width * height * sizeof(int));
    int *traceback = (int *)malloc(width * height * sizeof(int));

    // Initialize first row
    memcpy(dp, energy, width * sizeof(int));

    // Triangular-blocked dynamic programming
    for (int y = 1; y < height; y++) {
        #pragma omp parallel for schedule(static)
        for (int x = 0; x < width; x++) {
            int min_energy = dp[(y-1)*width + x];
            int min_x = x;

            if(x > 0 && dp[(y-1)*width + (x-1)] < min_energy) {
                min_energy = dp[(y-1)*width + (x-1)];
                min_x = x - 1;
            }
            if(x < width-1 && dp[(y-1)*width + (x+1)] < min_energy) {
                min_energy = dp[(y-1)*width + (x+1)];
                min_x = x + 1;
            }

            dp[y*width + x] = energy[y*width + x] + min_energy;
            traceback[y*width + x] = min_x;
        }
    }

    // Find minimum in last row with reduction
    int min_index = 0;
    int min_value = dp[(height-1)*width];
    #pragma omp parallel for reduction(min:min_value)
    for (int x = 1; x < width; x++) {
        if(dp[(height-1)*width + x] < min_value) {
            min_value = dp[(height-1)*width + x];
            min_index = x;
        }
    }

    // Traceback (sequential)
    seam[height-1] = min_index;
    for (int y = height-2; y >= 0; y--) {
        seam[y] = traceback[(y+1)*width + seam[y+1]];
    }

    free(dp);
    free(traceback);
}

// Function to remove multiple seams in parallel (greedy approach)
unsigned char *remove_multiple_seams(const unsigned char *image, int width, int height, const int *seams, int num_seams) {
    int new_width = width - num_seams;
    unsigned char *new_image = (unsigned char *)malloc(new_width * height * COLOR_CHANNELS);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        const unsigned char *src_row = image + y * width * COLOR_CHANNELS;
        unsigned char *dst_row = new_image + y * new_width * COLOR_CHANNELS;

        int dst_x = 0;
        for (int x = 0; x < width; x++) {
            int is_seam = 0;
            for (int s = 0; s < num_seams; s++) {
                if (seams[s * height + y] == x) {
                    is_seam = 1;
                    break;
                }
            }
            if (!is_seam) {
                memcpy(dst_row + dst_x * COLOR_CHANNELS, src_row + x * COLOR_CHANNELS, COLOR_CHANNELS);
                dst_x++;
            }
        }
    }

    return new_image;
}

// Function to log execution time
void log_execution_time(const char *filename, double elapsed_time) {
    struct stat st = {0};
    if (stat("parallel_results", &st) == -1) {
        mkdir("parallel_results", 0700);
    }

    // Get the number of threads and cores
    int num_threads = omp_get_max_threads();
    int num_cores = omp_get_num_procs();

    char filepath[256];
    snprintf(filepath, sizeof(filepath), "parallel_results/cores%d_threads%d_%s.txt", num_cores, num_threads, filename);

    FILE *file = fopen(filepath, "a");
    if (file) {
        fprintf(file, "%.4f\n", elapsed_time);
        fclose(file);
    } else {
        fprintf(stderr, "Error: Could not write to %s\n", filepath);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "USAGE: %s input_image output_image\n", argv[0]);
        return EXIT_FAILURE;
    }

    int width, height, channels;
    unsigned char *image = stbi_load(argv[1], &width, &height, &channels, COLOR_CHANNELS);
    if (!image) {
        fprintf(stderr, "Error: Could not load image %s\n", argv[1]);
        return EXIT_FAILURE;
    }

    int target_width, target_height;
    if (sscanf(argv[2], "%dx%d", &target_width, &target_height) != 2 || target_height != height) {
        fprintf(stderr, "Error: Output filename must specify widthxheight (e.g., 1800x1080.png)\n");
        stbi_image_free(image);
        return EXIT_FAILURE;
    }

    int num_seams = width - target_width;
    if (num_seams <= 0) {
        fprintf(stderr, "Error: Target width must be smaller than input width\n");
        stbi_image_free(image);
        return EXIT_FAILURE;
    }

    clock_t start_time = clock();

    // Preallocate reusable buffers
    int *energy = (int *)malloc(width * height * sizeof(int));
    int *seams = (int *)malloc(num_seams * height * sizeof(int));
    unsigned char *current_image = (unsigned char *)malloc(width * height * COLOR_CHANNELS);
    memcpy(current_image, image, width * height * COLOR_CHANNELS);

    int current_width = width;
    for (int i = 0; i < num_seams; i++) {
        compute_energy(current_image, current_width, height, energy);
        find_seam_triangular(energy, current_width, height, seams + i * height);

        unsigned char *temp_image = remove_multiple_seams(current_image, current_width, height, seams, i + 1);
        free(current_image);
        current_image = temp_image;
        current_width--;
    }

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    log_execution_time(argv[2], elapsed_time);

    // Ensure output directory exists
    struct stat st = {0};
    if (stat("parallel_results", &st) == -1) {
        mkdir("parallel_results", 0700);
    }

    char output_path[256];
    snprintf(output_path, sizeof(output_path), "parallel_results/%s", argv[2]);

    stbi_write_png(output_path, current_width, height, COLOR_CHANNELS, current_image, current_width * COLOR_CHANNELS);

    stbi_image_free(image);
    free(current_image);
    free(energy);
    free(seams);

    return 0;
}