#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define COLOR_CHANNELS 3  // RGB Images

// Compute Energy Map using Sobel Operator
void compute_energy(const unsigned char *image, int width, int height, int channels, int *energy) {
    int gx, gy;
    int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            gx = gy = 0;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int xi = x + j;
                    int yi = y + i;
                    if (xi < 0) xi = 0;
                    if (xi >= width) xi = width - 1;
                    if (yi < 0) yi = 0;
                    if (yi >= height) yi = height - 1;

                    int idx = (yi * width + xi) * channels;
                    int gray = (image[idx] + image[idx + 1] + image[idx + 2]) / 3;  

                    gx += gray * sobel_x[i + 1][j + 1];
                    gy += gray * sobel_y[i + 1][j + 1];
                }
            }
            energy[y * width + x] = abs(gx) + abs(gy);  // Absolute sum for energy
        }
    }
}

// Find the lowest-energy vertical seam
void find_seam(const int *energy, int width, int height, int *seam) {
    int *dp = (int *)malloc(width * height * sizeof(int));
    int *traceback = (int *)malloc(width * height * sizeof(int));

    // Initialize the first row
    for (int x = 0; x < width; x++) {
        dp[x] = energy[x];
    }

    // Compute the DP table
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

    // Trace back the lowest-energy seam
    int min_index = 0;
    for (int x = 1; x < width; x++) {
        if (dp[(height - 1) * width + x] < dp[(height - 1) * width + min_index]) {
            min_index = x;
        }
    }

    for (int y = height - 1; y >= 0; y--) {
        seam[y] = min_index;
        min_index = traceback[y * width + min_index];
    }

    free(dp);
    free(traceback);
}

// Remove a vertical seam correctly by shifting the image into a new buffer
unsigned char *remove_seam(const unsigned char *image, int width, int height, int channels, const int *seam) {
    int new_width = width - 1;
    unsigned char *new_image = (unsigned char *)malloc(new_width * height * channels);

    for (int y = 0; y < height; y++) {
        int new_x = 0;
        for (int x = 0; x < width; x++) {
            if (x == seam[y]) continue;  // Skip the seam pixel

            int old_idx = (y * width + x) * channels;
            int new_idx = (y * new_width + new_x) * channels;

            for (int c = 0; c < channels; c++) {
                new_image[new_idx + c] = image[old_idx + c];
            }
            new_x++;
        }
    }
    return new_image;
}

// Main Function
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("USAGE: %s input_image output_image\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Load the input image
    int width, height, channels;
    unsigned char *image = stbi_load(argv[1], &width, &height, &channels, COLOR_CHANNELS);
    if (!image) {
        printf("Error: Could not load image %s\n", argv[1]);
        return EXIT_FAILURE;
    }
    
    printf("Loaded image %s of size %dx%d\n", argv[1], width, height);

    // Get target width from output filename (e.g., "1800x1080.png")
    int target_width, target_height;
    if (sscanf(argv[2], "%dx%d", &target_width, &target_height) != 2 || target_height != height) {
        printf("Error: Output filename must specify widthxheight (e.g., 1800x1080.png)\n");
        stbi_image_free(image);
        return EXIT_FAILURE;
    }

    int num_seams = width - target_width;
    if (num_seams <= 0) {
        printf("Error: Target width must be smaller than input width\n");
        stbi_image_free(image);
        return EXIT_FAILURE;
    }

    unsigned char *new_image = (unsigned char *)malloc(width * height * channels);
    memcpy(new_image, image, width * height * channels);

    // Seam carving loop
    for (int i = 0; i < num_seams; i++) {
        int *energy = (int *)malloc(width * height * sizeof(int));
        int *seam = (int *)malloc(height * sizeof(int));

        compute_energy(new_image, width, height, channels, energy);
        find_seam(energy, width, height, seam);

        unsigned char *temp_image = remove_seam(new_image, width, height, channels, seam);
        free(new_image);
        new_image = temp_image;

        free(energy);
        free(seam);
        width--;
    }

    // Save the output image
    stbi_write_png(argv[2], width, height, channels, new_image, width * channels);
    printf("Resized image saved to %s\n", argv[2]);

    stbi_image_free(image);
    free(new_image);
    return 0;
}
