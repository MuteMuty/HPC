/* #include <stdio.h>
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

void find_seam(const int *energy, int width, int height, int *seam) {
    int *dp = (int *)malloc(width * height * sizeof(int));
    int *traceback = (int *)malloc(width * height * sizeof(int));

    // Initialize first row
    memcpy(dp, energy, width * sizeof(int));

    // Dynamic programming with row-wise parallelism
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

unsigned char *remove_seam(const unsigned char *image, int width, int height, const int *seam) {
    int new_width = width - 1;
    unsigned char *new_image = (unsigned char *)malloc(new_width * height * COLOR_CHANNELS);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        int seam_x = seam[y];
        const unsigned char *src_row = image + y * width * COLOR_CHANNELS;
        unsigned char *dst_row = new_image + y * new_width * COLOR_CHANNELS;

        // Split memcpy into two parts for better parallelism
        if(seam_x > 0) {
            memcpy(dst_row, src_row, seam_x * COLOR_CHANNELS);
        }
        if(seam_x < new_width) {
            memcpy(dst_row + seam_x * COLOR_CHANNELS,
                   src_row + (seam_x + 1) * COLOR_CHANNELS,
                   (new_width - seam_x) * COLOR_CHANNELS);
        }
    }

    return new_image;
}

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
    int *seam = (int *)malloc(height * sizeof(int));
    unsigned char *current_image = (unsigned char *)malloc(width * height * COLOR_CHANNELS);
    memcpy(current_image, image, width * height * COLOR_CHANNELS);

    int current_width = width;
    for (int i = 0; i < num_seams; i++) {
        compute_energy(current_image, current_width, height, energy);
        find_seam(energy, current_width, height, seam);
        
        unsigned char *temp_image = remove_seam(current_image, current_width, height, seam);
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
    free(seam);

    return 0;
} */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include <omp.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 3

void compute_energy(const unsigned char *image, int width, int height, int *energy) {
    #pragma omp parallel for schedule(static, 16)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const unsigned char *pixel = image + (y * width + x) * COLOR_CHANNELS;
            
            int left_x = (x > 0) ? x - 1 : x;
            int right_x = (x < width - 1) ? x + 1 : x;
            const unsigned char *left = image + (y * width + left_x) * COLOR_CHANNELS;
            const unsigned char *right = image + (y * width + right_x) * COLOR_CHANNELS;
            int dx = abs(left[0] - right[0]) + abs(left[1] - right[1]) + abs(left[2] - right[2]);

            int top_y = (y > 0) ? y - 1 : y;
            int bottom_y = (y < height - 1) ? y + 1 : y;
            const unsigned char *top = image + (top_y * width + x) * COLOR_CHANNELS;
            const unsigned char *bottom = image + (bottom_y * width + x) * COLOR_CHANNELS;
            int dy = abs(top[0] - bottom[0]) + abs(top[1] - bottom[1]) + abs(top[2] - bottom[2]);

            energy[y * width + x] = dx + dy;
        }
    }
}

void find_seam(const int *energy, int width, int height, int *seam) {
    int *dp = (int *)malloc(width * height * sizeof(int));
    int *traceback = (int *)malloc(width * height * sizeof(int));

    memcpy(dp, energy, width * sizeof(int));

    for (int y = 1; y < height; y++) {
        #pragma omp parallel for schedule(static)
        for (int x = 0; x < width; x++) {
            int min_energy = dp[(y-1) * width + x];
            int min_x = x;

            if (x > 0 && dp[(y-1) * width + (x-1)] < min_energy) {
                min_energy = dp[(y-1) * width + (x-1)];
                min_x = x - 1;
            }
            if (x < width - 1 && dp[(y-1) * width + (x+1)] < min_energy) {
                min_energy = dp[(y-1) * width + (x+1)];
                min_x = x + 1;
            }

            dp[y * width + x] = energy[y * width + x] + min_energy;
            traceback[y * width + x] = min_x;
        }
    }

    int min_index = 0;
    int min_value = dp[(height-1) * width];

    #pragma omp parallel
    {
        int local_min_index = 0;
        int local_min_value = dp[(height-1) * width];

        #pragma omp for nowait
        for (int x = 1; x < width; x++) {
            if (dp[(height-1) * width + x] < local_min_value) {
                local_min_value = dp[(height-1) * width + x];
                local_min_index = x;
            }
        }

        #pragma omp critical
        {
            if (local_min_value < min_value) {
                min_value = local_min_value;
                min_index = local_min_index;
            }
        }
    }

    seam[height-1] = min_index;
    for (int y = height - 2; y >= 0; y--) {
        seam[y] = traceback[(y+1) * width + seam[y+1]];
    }

    free(dp);
    free(traceback);
}

void remove_seam(unsigned char *image, int width, int height, const int *seam) {
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < height; y++) {
        int new_idx = 0;
        for (int x = 0; x < width; x++) {
            if (x == seam[y]) continue;
            int old_pos = (y * width + x) * COLOR_CHANNELS;
            int new_pos = (y * (width - 1) + new_idx) * COLOR_CHANNELS;
            memcpy(&image[new_pos], &image[old_pos], COLOR_CHANNELS);
            new_idx++;
        }
    }
}

void log_execution_time(const char *filename, double elapsed_time) {
    int num_threads = omp_get_max_threads();
    int num_cores = omp_get_num_procs();

    FILE *file = fopen("parallel_results/execution_times.txt", "a");
    if (file) {
        fprintf(file, "Cores: %d, Threads: %d, Image: %s, Time: %.4f sec\n", num_cores, num_threads, filename, elapsed_time);
        fclose(file);
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
        fprintf(stderr, "Error: Output filename must specify widthxheight\n");
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

    int *energy = (int *)malloc(width * height * sizeof(int));
    int *seam = (int *)malloc(height * sizeof(int));
    
    unsigned char *current_image = (unsigned char *)malloc(width * height * COLOR_CHANNELS);
    memcpy(current_image, image, width * height * COLOR_CHANNELS);
    
    int current_width = width;
    for (int i = 0; i < num_seams; i++) {
        compute_energy(current_image, current_width, height, energy);
        find_seam(energy, current_width, height, seam);
        remove_seam(current_image, current_width, height, seam);
        current_width--;
    }

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    log_execution_time(argv[2], elapsed_time);

    mkdir("parallel_results", 0700);
    char output_path[256];
    snprintf(output_path, sizeof(output_path), "parallel_results/%s", argv[2]);
    stbi_write_png(output_path, current_width, height, COLOR_CHANNELS, current_image, current_width * COLOR_CHANNELS);

    stbi_image_free(image);
    free(current_image);
    free(energy);
    free(seam);

    return 0;
}
