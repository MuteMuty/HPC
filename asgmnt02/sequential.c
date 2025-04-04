#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

void histogram_equalization(unsigned char* image, int width, int height, int channels) {
    int total_pixels = width * height;

    // Allocate memory for Y, U, V
    unsigned char* Y = (unsigned char*)malloc(total_pixels);
    float* U = (float*)malloc(total_pixels * sizeof(float));
    float* V = (float*)malloc(total_pixels * sizeof(float));

    // Convert RGB to YUV
    for (int i = 0; i < total_pixels; i++) {
        unsigned char R = image[i * channels];
        unsigned char G = image[i * channels + 1];
        unsigned char B = image[i * channels + 2];

        // Compute Y (luminance)
        float Y_float = 0.299f * R + 0.587f * G + 0.114f * B;
        Y[i] = (unsigned char)(Y_float + 0.5f); // Round to nearest integer

        // Compute U and V (chrominance)
        U[i] = (-0.168736f * R) + (-0.331264f * G) + (0.5f * B) + 128.0f;
        V[i] = (0.5f * R) + (-0.418688f * G) + (-0.081312f * B) + 128.0f;
    }

    // Compute histogram Hy
    int Hy[256] = {0};
    for (int i = 0; i < total_pixels; i++) {
        Hy[Y[i]]++;
    }

    // Compute cumulative histogram H_cdf
    int H_cdf[256];
    H_cdf[0] = Hy[0];
    for (int i = 1; i < 256; i++) {
        H_cdf[i] = H_cdf[i - 1] + Hy[i];
    }

    // Find minimum non-zero value in H_cdf
    int min_cdf = 0;
    for (int l = 0; l < 256; l++) {
        if (H_cdf[l] > 0) {
            min_cdf = H_cdf[l];
            break;
        }
    }

    // Compute lookup table (LUT)
    int LUT[256];
    int denominator = total_pixels - min_cdf;
    if (denominator == 0) {
        // All pixels have the same value
        for (int l = 0; l < 256; l++) {
            LUT[l] = (Hy[l] > 0) ? 255 : 0;
        }
    } else {
        float scale = 255.0f / denominator;
        for (int l = 0; l < 256; l++) {
            if (H_cdf[l] == 0) {
                LUT[l] = 0;
            } else {
                float normalized = (H_cdf[l] - min_cdf) * scale;
                LUT[l] = (int)floorf(normalized);
                // Clamp to 0-255
                LUT[l] = LUT[l] < 0 ? 0 : (LUT[l] > 255 ? 255 : LUT[l]);
            }
        }
    }

    // Apply LUT to Y and convert back to RGB
    for (int i = 0; i < total_pixels; i++) {
        unsigned char Y_new = LUT[Y[i]];

        // Compute adjusted U and V (subtract 128)
        float U_adj = U[i] - 128.0f;
        float V_adj = V[i] - 128.0f;

        // Convert YUV to RGB
        float R = Y_new + 1.402f * V_adj;
        float G = Y_new - 0.344136f * U_adj - 0.714136f * V_adj;
        float B = Y_new + 1.772f * U_adj;

        // Clamp and round RGB values
        R = fmaxf(0.0f, fminf(R, 255.0f));
        G = fmaxf(0.0f, fminf(G, 255.0f));
        B = fmaxf(0.0f, fminf(B, 255.0f));

        image[i * channels]     = (unsigned char)(R + 0.5f);
        image[i * channels + 1] = (unsigned char)(G + 0.5f);
        image[i * channels + 2] = (unsigned char)(B + 0.5f);
    }

    free(Y);
    free(U);
    free(V);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <input_image> <output_image>\n", argv[0]);
        return -1;
    }

    const char* input_path = argv[1];
    const char* output_path = argv[2];

    int width, height, channels;
    unsigned char* image = stbi_load(input_path, &width, &height, &channels, 3);
    if (!image) {
        fprintf(stderr, "Error loading image %s\n", input_path);
        return -1;
    }

    clock_t start = clock();
    histogram_equalization(image, width, height, 3);
    clock_t end = clock();
    
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000; // in milliseconds

    // Print timing and dimension information for parsing
    printf("DIMS %d %d TIME %.3f\n", width, height, time_taken);

    // Determine output format
    const char* ext = strrchr(output_path, '.');
    if (!ext) {
        fprintf(stderr, "Output file must have an extension\n");
        stbi_image_free(image);
        return -1;
    }
    ext++; // Skip the dot

    int success;
    if (strcasecmp(ext, "png") == 0) {
        success = stbi_write_png(output_path, width, height, 3, image, width * 3);
    } else if (strcasecmp(ext, "jpg") == 0 || strcasecmp(ext, "jpeg") == 0) {
        success = stbi_write_jpg(output_path, width, height, 3, image, 100);
    } else if (strcasecmp(ext, "bmp") == 0) {
        success = stbi_write_bmp(output_path, width, height, 3, image);
    } else {
        fprintf(stderr, "Unsupported format: %s\n", ext);
        stbi_image_free(image);
        return -1;
    }

    if (!success) {
        fprintf(stderr, "Error writing image to %s\n", output_path);
        stbi_image_free(image);
        return -1;
    }

    stbi_image_free(image);
    printf("Histogram equalization completed. Output saved to %s\n", output_path);
    return 0;
}