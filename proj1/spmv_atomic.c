#include "genresult.cuh"
#include <sys/time.h>

__global__ void getMulAtomic_kernel(MatrixInfo *mat, MatrixInfo *vec, MatrixInfo * res){
    
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // col tid
    int thread_num = blockDim.x * gridDim.x;
    int iter = mat->nz % thread_num ? mat->nz / thread_num + 1 : mat->nz / thread_num;

    for (int i = 0; i < iter; i++) {
        int dataid = thread_id + i * thread_num;
        if (dataid < mat->nz) {
            float data = mat->val[dataid];
            int row = mat->rIndex[dataid];
            int col = mat->cIndex[dataid];
            float temp = data * vec->val[col];
            atomicAdd(&res->val[row], temp);
        }
    }
}

size_t matrixSize(MatrixInfo * mat) {
    int intSize = sizeof(int);
    int floatSize = sizeof(float);
    
    int ints = intSize * 3; // M, N, nz
    int arrays = (mat->nz) * intSize * 2; // rIndex, cIndex
    int vals = (mat->nz) * floatSize; 

    return ints + arrays + vals;

}

void getMulAtomic(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    // allocate
    MatrixInfo * d_mat;
    MatrixInfo * d_vec;
    MatrixInfo * d_res;
    cudaMalloc((void**) &d_mat, matrixSize(mat));
    cudaMalloc((void**) &d_vec, matrixSize(vec));
    cudaMalloc((void**) &d_res, matrixSize(vec)); // resulting vector will always be size of constituent vector
    cudaMemcpy(d_mat, mat, matrixSize(mat), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, vec, matrixSize(vec), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, res, matrixSize(vec), cudaMemcpyHostToDevice);


    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    /*Invoke kernels...*/
    getMulAtomic_kernel<<<blockNum, blockSize>>>(d_mat, d_vec, d_res);

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Atomic Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);
   
    /* map nz's back to original form */
    cudaMemcpy(res, d_res, matrixSize(vec), cudaMemcpyDeviceToHost);
    
    /* Deallocate, please */
    cudaFree(d_mat);
    cudaFree(d_vec);
    cudaFree(d_res);
}
