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
#define BLOCK_WIDTH 64
#define SEAM_BATCH 16

typedef struct {
    int width;
    int height;
    unsigned char *data;
} Image;

// Triangular-blocked energy computation
void compute_energy(const Image *img, int *energy) {
    const int w = img->width;
    const int h = img->height;
    const unsigned char *image = img->data;

    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int dx, dy;
            
            // Horizontal gradient
            int left = (x > 0) ? x - 1 : x;
            int right = (x < w-1) ? x + 1 : x;
            const unsigned char *l = &image[(y*w + left)*COLOR_CHANNELS];
            const unsigned char *r = &image[(y*w + right)*COLOR_CHANNELS];
            dx = abs(l[0]-r[0]) + abs(l[1]-r[1]) + abs(l[2]-r[2]);

            // Vertical gradient
            int top = (y > 0) ? y - 1 : y;
            int bottom = (y < h-1) ? y + 1 : y;
            const unsigned char *t = &image[(top*w + x)*COLOR_CHANNELS];
            const unsigned char *b = &image[(bottom*w + x)*COLOR_CHANNELS];
            dy = abs(t[0]-b[0]) + abs(t[1]-b[1]) + abs(t[2]-b[2]);

            energy[y*w + x] = dx + dy;
        }
    }
}

// Triangular-blocked DP for cumulative energy
void find_seams(const int *energy, int width, int height, int *seams, int num_seams) {
    int *dp = (int *)aligned_alloc(64, width*height*sizeof(int));
    int *traceback = (int *)aligned_alloc(64, width*height*sizeof(int));

    // Initialize first row
    memcpy(dp, energy, width*sizeof(int));

    // Process in triangular blocks
    for (int y = 1; y < height; y++) {
        // Phase 1: Downward triangles
        #pragma omp parallel for schedule(dynamic, BLOCK_WIDTH)
        for (int bx = 0; bx < width; bx += BLOCK_WIDTH) {
            int start = bx;
            int end = (bx + BLOCK_WIDTH) < width ? bx + BLOCK_WIDTH : width;
            
            for (int x = start; x < end; x++) {
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

        // Phase 2: Upward triangles (dependency resolution)
        #pragma omp parallel for schedule(dynamic, BLOCK_WIDTH)
        for (int bx = BLOCK_WIDTH/2; bx < width; bx += BLOCK_WIDTH) {
            int start = bx;
            int end = (bx + BLOCK_WIDTH) < width ? bx + BLOCK_WIDTH : width;
            
            for (int x = start; x < end; x++) {
                if(x > 0 && dp[y*width + (x-1)] + energy[y*width + x] < dp[y*width + x]) {
                    dp[y*width + x] = dp[y*width + (x-1)] + energy[y*width + x];
                    traceback[y*width + x] = x - 1;
                }
                if(x < width-1 && dp[y*width + (x+1)] + energy[y*width + x] < dp[y*width + x]) {
                    dp[y*width + x] = dp[y*width + (x+1)] + energy[y*width + x];
                    traceback[y*width + x] = x + 1;
                }
            }
        }
    }

    // Find multiple seams using parallel reduction
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, SEAM_BATCH)
        for (int s = 0; s < num_seams; s++) {
            int min_idx = 0;
            int min_val = dp[(height-1)*width];
            
            // Find local minimum in last row
            for (int x = 1; x < width; x++) {
                if(dp[(height-1)*width + x] < min_val) {
                    min_val = dp[(height-1)*width + x];
                    min_idx = x;
                }
            }

            // Traceback
            seams[s*height + height-1] = min_idx;
            for (int y = height-2; y >= 0; y--) {
                seams[s*height + y] = traceback[(y+1)*width + seams[s*height + y+1]];
            }
        }
    }

    free(dp);
    free(traceback);
}

// Comparison function for sorting seams
int compare_desc(const void *a, const void *b) {
    return (*(int*)b - *(int*)a);
}

// Parallel seam removal with correct ordering
Image remove_seams(Image img, const int *seams, int num_seams) {
    int new_width = img.width - num_seams;
    unsigned char *new_data = (unsigned char *)aligned_alloc(64, new_width * img.height * COLOR_CHANNELS);
    
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < img.height; y++) {
        int *row_seams = (int *)alloca(num_seams * sizeof(int));
        for (int s = 0; s < num_seams; s++) {
            row_seams[s] = seams[s * img.height + y];
        }
        qsort(row_seams, num_seams, sizeof(int), compare_desc);

        int src_idx = 0;
        int dst_idx = 0;
        for (int s = 0; s < num_seams; s++) {
            // Copy pixels before seam
            while(src_idx < row_seams[s]) {
                memcpy(&new_data[(y * new_width + dst_idx) * COLOR_CHANNELS],
                       &img.data[(y * img.width + src_idx) * COLOR_CHANNELS],
                       COLOR_CHANNELS);
                dst_idx++;
                src_idx++;
            }
            // Skip the seam pixel
            src_idx++;
        }
        // Copy remaining pixels
        while(src_idx < img.width) {
            memcpy(&new_data[(y * new_width + dst_idx) * COLOR_CHANNELS],
                   &img.data[(y * img.width + src_idx) * COLOR_CHANNELS],
                   COLOR_CHANNELS);
            dst_idx++;
            src_idx++;
        }
    }

    free(img.data);
    return (Image){new_width, img.height, new_data};
}

void log_execution_time(const char *filename, double elapsed_time) {
    int num_threads = omp_get_max_threads();
    int num_cores = omp_get_num_procs();

    FILE *file = fopen("bonus_results/execution_times.txt", "a");
    if (file) {
        fprintf(file, "Cores: %d, Threads: %d, Image: %s, Time: %.4f sec\n", num_cores, num_threads, filename, elapsed_time);
        fclose(file);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input output\n", argv[0]);
        return 1;
    }

    // Load image
    int width, height, channels;
    unsigned char *data = stbi_load(argv[1], &width, &height, &channels, COLOR_CHANNELS);
    Image img = {width, height, data};

    // Parse target size
    int target_width, target_height;
    if (sscanf(argv[2], "%dx%d", &target_width, &target_height) != 2 || target_height != height) {
        fprintf(stderr, "Invalid output format\n");
        return 1;
    }

    const int num_seams = width - target_width;
    int *seams = (int *)aligned_alloc(64, num_seams*height*sizeof(int));

    double start_time = omp_get_wtime();

    // Process in batches for better cache utilization
    for (int processed = 0; processed < num_seams; processed += SEAM_BATCH) {
        int batch_size = (num_seams - processed) > SEAM_BATCH 
                    ? SEAM_BATCH : (num_seams - processed);
        
        int *energy = (int*)aligned_alloc(64, img.width*img.height*sizeof(int));
        compute_energy(&img, energy);
        
        int *current_seams = (int*)aligned_alloc(64, batch_size*img.height*sizeof(int));
        find_seams(energy, img.width, img.height, current_seams, batch_size);
        
        Image new_img = remove_seams(img, current_seams, batch_size);
        
        // Update image reference AFTER successful removal
        free(energy);
        free(current_seams);
        img = new_img;  // Old data freed inside remove_seams
    }

    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    log_execution_time(argv[2], elapsed_time);

    // Save result
    char output_path[256];
    snprintf(output_path, sizeof(output_path), "bonus_results/%s", argv[2]);
    stbi_write_png(output_path, img.width, img.height, COLOR_CHANNELS, img.data, img.width*COLOR_CHANNELS);

    free(seams);
    free(img.data);
    return 0;
}