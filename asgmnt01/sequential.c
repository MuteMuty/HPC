#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 3

void compute_energy(const unsigned char *image, int width, int height, int *energy) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Get current pixel
            const unsigned char *pixel = image + (y * width + x) * COLOR_CHANNELS;

            // Get neighboring pixels (clamped to boundaries)
            int left_x = (x > 0) ? x - 1 : x;
            int right_x = (x < width - 1) ? x + 1 : x;
            int top_y = (y > 0) ? y - 1 : y;
            int bottom_y = (y < height - 1) ? y + 1 : y;

            const unsigned char *left_pixel = image + (y * width + left_x) * COLOR_CHANNELS;
            const unsigned char *right_pixel = image + (y * width + right_x) * COLOR_CHANNELS;
            const unsigned char *top_pixel = image + (top_y * width + x) * COLOR_CHANNELS;
            const unsigned char *bottom_pixel = image + (bottom_y * width + x) * COLOR_CHANNELS;

            // Compute horizontal and vertical gradients
            int dx_r = abs(left_pixel[0] - right_pixel[0]);
            int dx_g = abs(left_pixel[1] - right_pixel[1]);
            int dx_b = abs(left_pixel[2] - right_pixel[2]);
            int horizontal_gradient = dx_r + dx_g + dx_b;

            int dy_r = abs(top_pixel[0] - bottom_pixel[0]);
            int dy_g = abs(top_pixel[1] - bottom_pixel[1]);
            int dy_b = abs(top_pixel[2] - bottom_pixel[2]);
            int vertical_gradient = dy_r + dy_g + dy_b;

            // Total energy is the sum of horizontal and vertical gradients
            energy[y * width + x] = horizontal_gradient + vertical_gradient;
        }
    }
}

void find_seam(const int *energy, int width, int height, int *seam) {
    int *dp = (int *)malloc(width * height * sizeof(int));
    int *traceback = (int *)malloc(width * height * sizeof(int));

    // Initialize first row
    memcpy(dp, energy, width * sizeof(int));

    for (int y = 1; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int min_energy = dp[(y - 1) * width + x];
            int min_x = x;

            if (x > 0 && dp[(y - 1) * width + (x - 1)] < min_energy) {
                min_energy = dp[(y - 1) * width + (x - 1)];
                min_x = x - 1;
            }
            if (x < width - 1 && dp[(y - 1) * width + (x + 1)] < min_energy) {
                min_energy = dp[(y - 1) * width + (x + 1)];
                min_x = x + 1;
            }

            dp[y * width + x] = energy[y * width + x] + min_energy;
            traceback[y * width + x] = min_x;
        }
    }

    // Find the minimum in the last row
    int min_index = 0;
    for (int x = 1; x < width; x++) {
        if (dp[(height - 1) * width + x] < dp[(height - 1) * width + min_index]) {
            min_index = x;
        }
    }

    // Traceback to find the seam
    seam[height - 1] = min_index;
    for (int y = height - 2; y >= 0; y--) {
        seam[y] = traceback[(y + 1) * width + seam[y + 1]];
    }

    free(dp);
    free(traceback);
}

unsigned char *remove_seam(const unsigned char *image, int width, int height, const int *seam) {
    int new_width = width - 1;
    unsigned char *new_image = (unsigned char *)malloc(new_width * height * COLOR_CHANNELS);

    for (int y = 0; y < height; y++) {
        int seam_x = seam[y];
        const unsigned char *src = image + y * width * COLOR_CHANNELS;
        unsigned char *dst = new_image + y * new_width * COLOR_CHANNELS;

        if (seam_x > 0) {
            memcpy(dst, src, seam_x * COLOR_CHANNELS);
        }
        if (seam_x < new_width) {
            memcpy(dst + seam_x * COLOR_CHANNELS, 
                   src + (seam_x + 1) * COLOR_CHANNELS, 
                   (new_width - seam_x) * COLOR_CHANNELS);
        }
    }

    return new_image;
}

void log_execution_time(const char *filename, double elapsed_time) {
    struct stat st = {0};
    if (stat("sequential_results", &st) == -1) {
        mkdir("sequential_results", 0700);
    }

    char filepath[256];
    snprintf(filepath, sizeof(filepath), "sequential_results/col_diff_grad_%s.txt", filename);

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

    // Preallocate energy and seam buffers
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
    if (stat("sequential_results", &st) == -1) {
        mkdir("sequential_results", 0700);
    }

    char output_path[256];
    snprintf(output_path, sizeof(output_path), "sequential_results/%s", argv[2]);

    stbi_write_png(output_path, current_width, height, COLOR_CHANNELS, current_image, current_width * COLOR_CHANNELS);

    stbi_image_free(image);
    free(current_image);
    free(energy);
    free(seam);

    return 0;
}