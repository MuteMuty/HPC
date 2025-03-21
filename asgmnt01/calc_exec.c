#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 100

// Structure to store sum and count for each group
typedef struct {
    int cores;
    int threads;
    char image[20];
    double sum;
    int count;
} Group;

// Function to find or create a group
Group* find_or_create_group(Group* groups, int* num_groups, int cores, int threads, const char* image) {
    for (int i = 0; i < *num_groups; i++) {
        if (groups[i].cores == cores && groups[i].threads == threads && strcmp(groups[i].image, image) == 0) {
            return &groups[i];
        }
    }
    // If not found, create a new group
    groups[*num_groups].cores = cores;
    groups[*num_groups].threads = threads;
    strcpy(groups[*num_groups].image, image);
    groups[*num_groups].sum = 0;
    groups[*num_groups].count = 0;
    (*num_groups)++;
    return &groups[*num_groups - 1];
}

int main() {
    FILE* file = fopen("parallel_results/execution_times.txt", "r");
    if (!file) {
        perror("Failed to open file");
        return 1;
    }

    Group groups[100]; // Adjust size as needed
    int num_groups = 0;

    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), file)) {
        int cores, threads;
        char image[20];
        double time;

        // Parse the line
        sscanf(line, "Cores: %d, Threads: %d, Image: %[^,], Time: %lf sec", &cores, &threads, image, &time);

        // Find or create the group
        Group* group = find_or_create_group(groups, &num_groups, cores, threads, image);

        // Update the sum and count
        group->sum += time;
        group->count++;
    }

    fclose(file);

    // Print the averages
    printf("Cores, Threads, Image, Average Time (sec)\n");
    for (int i = 0; i < num_groups; i++) {
        double average = groups[i].sum / groups[i].count;
        printf("%d, %d, %s, %.4f\n", groups[i].cores, groups[i].threads, groups[i].image, average);
    }

    return 0;
}