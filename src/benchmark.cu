#include "helper_cuda.h"
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VALUES 256
#define BANKS 32

void *allocate_memory(size_t size) {
    void *data = malloc(size);

    if (data == NULL) {
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }

    return data;
}

__global__ void histogram(unsigned char *data, int *positions, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    __shared__ int counts[VALUES];

    counts[threadIdx.x] = 0;
    __syncthreads();

    for (int i = idx; i < n; i += offset) {
        atomicAdd(&counts[data[i]], 1);
    }

    __syncthreads();
    atomicAdd(&positions[threadIdx.x], counts[threadIdx.x]);
}

__global__ void scan(int *positions, int n) {
    int idx = threadIdx.x;
    int offset = 1;
    int left, right, tmp;

    __shared__ int values[VALUES + VALUES / BANKS];

    values[idx + idx / BANKS] = positions[idx];
    __syncthreads();

    for (int size = VALUES / 2; size > 0; size /= 2) {
        if (idx < size) {
            left = offset * (2 * idx + 1) - 1;
            right = offset * (2 * idx + 2) - 1;
            values[right + right / BANKS] += values[left + left / BANKS];
        }

        offset *= 2;
        __syncthreads();
    }

    if (idx == 0) {
        values[VALUES - 1 + (VALUES - 1) / BANKS] = 0;
    }

    __syncthreads();

    for (int size = 1; size < VALUES; size *= 2) {
        offset /= 2;

        if (idx < size) {
            left = offset * (2 * idx + 1) - 1;
            right = offset * (2 * idx + 2) - 1;
            left += left / BANKS;
            right += right / BANKS;

            tmp = values[left];
            values[left] = values[right];
            values[right] += tmp;
        }

        __syncthreads();
    }

    positions[idx] = values[idx + idx / BANKS];

    if (idx == 0) {
        positions[VALUES] = n;
    }
}

__global__ void kernel(unsigned char *data, int *positions) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    int value = blockIdx.y;

    for (int i = positions[value] + idx; i < positions[value + 1]; i += offset) {
        data[i] = (unsigned char)value;
    }
}

void cpu_sort(unsigned char *data, unsigned char *result, int n) {
    int counts[VALUES] = {0};
    int position = 0;

    for (int i = 0; i < n; i++) {
        counts[data[i]]++;
    }

    for (int value = 0; value < VALUES; value++) {
        memset(result + position, value, counts[value]);
        position += counts[value];
    }
}

void run_test(int n, int repeats) {
    unsigned char *data = (unsigned char *)allocate_memory(n);
    unsigned char *cpu_result = (unsigned char *)allocate_memory(n);
    unsigned char *gpu_result = (unsigned char *)allocate_memory(n);

    for (int i = 0; i < n; i++) {
        data[i] = (unsigned char)(rand() % VALUES);
    }

    unsigned char *dev_data;
    int *dev_positions;

    checkCudaErrors(cudaMalloc(&dev_data, n));
    checkCudaErrors(cudaMalloc(&dev_positions, sizeof(int) * (VALUES + 1)));
    checkCudaErrors(cudaMemcpy(dev_data, data, n, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(dev_positions, 0, sizeof(int) * VALUES));

    histogram<<<256, VALUES>>>(dev_data, dev_positions, n);
    checkCudaErrors(cudaGetLastError());

    scan<<<1, VALUES>>>(dev_positions, n);
    checkCudaErrors(cudaGetLastError());

    kernel<<<dim3(32, VALUES), 256>>>(dev_data, dev_positions);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cudaEvent_t start, finish;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&finish));

    double gpu_time = 0.0;
    float elapsed;

    for (int repeat = 0; repeat < repeats; repeat++) {
        checkCudaErrors(cudaMemcpy(dev_data, data, n, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaEventRecord(start));
        checkCudaErrors(cudaMemset(dev_positions, 0, sizeof(int) * VALUES));
    
        histogram<<<256, VALUES>>>(dev_data, dev_positions, n);
        checkCudaErrors(cudaGetLastError());
    
        scan<<<1, VALUES>>>(dev_positions, n);
        checkCudaErrors(cudaGetLastError());
    
        kernel<<<dim3(32, VALUES), 256>>>(dev_data, dev_positions);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaEventRecord(finish));
        checkCudaErrors(cudaEventSynchronize(finish));
        checkCudaErrors(cudaEventElapsedTime(&elapsed, start, finish));
        gpu_time += elapsed;
    }

    gpu_time /= repeats;
    checkCudaErrors(cudaMemcpy(gpu_result, dev_data, n, cudaMemcpyDeviceToHost));

    auto cpu_start = std::chrono::steady_clock::now();

    for (int repeat = 0; repeat < repeats; repeat++) {
        cpu_sort(data, cpu_result, n);
    }

    auto cpu_finish = std::chrono::steady_clock::now();
    double cpu_time =
        std::chrono::duration<double, std::milli>(cpu_finish - cpu_start).count() /
        repeats;

    if (memcmp(cpu_result, gpu_result, n) != 0) {
        fprintf(stderr, "CPU and GPU results differ\n");
        exit(EXIT_FAILURE);
    }

    printf("n=%d GPU=%.6f ms CPU=%.6f ms speedup=%.6f\n", n, gpu_time,
           cpu_time, cpu_time / gpu_time);

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(finish));
    checkCudaErrors(cudaFree(dev_data));
    checkCudaErrors(cudaFree(dev_positions));
    free(data);
    free(cpu_result);
    free(gpu_result);
}

int main() {
    int sizes[] = {100000, 1000000, 10000000, 100000000};

    srand(42);

    for (int i = 0; i < 4; i++) {
        run_test(sizes[i], 5);
    }

    return 0;
}
