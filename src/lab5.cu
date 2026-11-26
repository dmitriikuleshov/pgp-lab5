#include <stdio.h>
#include <stdlib.h>

#define checkCudaErrors(call)                                                  \
    do {                                                                       \
        cudaError_t res = call;                                                \
        if (res != cudaSuccess) {                                              \
            fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__,         \
                    __LINE__, cudaGetErrorString(res));                        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define VALUES 256
#define BANKS 32
#define MAX_SIZE 537000000

void read_file(void *data, size_t size, size_t count, FILE *fp) {
    if (fread(data, size, count, fp) != count) {
        fprintf(stderr, "Failed to read file\n");
        exit(EXIT_FAILURE);
    }
}

void write_file(const void *data, size_t size, size_t count, FILE *fp) {
    if (fwrite(data, size, count, fp) != count) {
        fprintf(stderr, "Failed to write file\n");
        exit(EXIT_FAILURE);
    }
}

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
    // bank = index_in_values % 32
    // 256 elements + 8 additional elements
    // Elements: [ 0 ... 31 ]  f  [ 32 ... 63    ]  f  [ 64 ... 95 ]
    // Banks:    [ 0 ... 31 ]  0  [ 1  ... 31, 0 ]  1  [ 2 ... 31,0,1 ]

    // Elements: [ 0 ... 31 ]  [ 32 ... 63 ]  [ 64 ... 95 ]
    // Banks:    [ 0 ... 31 ]  [  0 ... 31 ]  [  0 ... 31 ]

    __shared__ int values[VALUES + VALUES / BANKS];

    // Нить 0:   positions[0]  → values[0]
    // Нить 31:  positions[31] → values[31] 
    //      values[32] — пропуск
    // Нить 32:  positions[32] → values[33]
    // Нить 33:  positions[33] → values[34]
    values[idx + idx / BANKS] = positions[idx];
    __syncthreads();

    // сумма каждой группы записывается в ее поледний элемент
    // на каждой итерации используется в 2 раза меньше нитей
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

    for (int i = positions[value] + idx; i < positions[value + 1];
         i += offset) {
        data[i] = (unsigned char)value;
    }
}

int main() {
    int n;
    read_file(&n, sizeof(int), 1, stdin);

    if (n < 0 || n > MAX_SIZE) {
        fprintf(stderr, "Invalid array size\n");
        return EXIT_FAILURE;
    }

    if (n == 0) {
        return 0;
    }

    size_t data_size = sizeof(unsigned char) * n;
    unsigned char *data = (unsigned char *)allocate_memory(data_size);
    read_file(data, sizeof(unsigned char), n, stdin);

    unsigned char *dev_data;
    int *dev_positions;

    checkCudaErrors(cudaMalloc(&dev_data, data_size));
    checkCudaErrors(cudaMalloc(&dev_positions, sizeof(int) * (VALUES + 1)));
    checkCudaErrors(
        cudaMemcpy(dev_data, data, data_size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(dev_positions, 0, sizeof(int) * VALUES));

    histogram<<<256, VALUES>>>(dev_data, dev_positions, n);
    checkCudaErrors(cudaGetLastError());

    scan<<<1, VALUES>>>(dev_positions, n);
    checkCudaErrors(cudaGetLastError());

    kernel<<<dim3(32, VALUES), 256>>>(dev_data, dev_positions);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(
        cudaMemcpy(data, dev_data, data_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(dev_data));
    checkCudaErrors(cudaFree(dev_positions));

    write_file(data, sizeof(unsigned char), n, stdout);
    free(data);

    return 0;
}
