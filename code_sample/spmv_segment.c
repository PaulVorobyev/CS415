#include "genresult.cuh"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>

struct Nz_{
  float val;
  int r;
  int c;
};
typedef struct Nz_ Nz;

__global__ void putProduct(int *len, int *r, float *nz, float *out, float* buffer){
  int i;
  /* get chunk size - last block gets the leftover elements */
  int avgChunkSize = *len / gridDim.x;
  int chunkSize = (blockIdx.x == gridDim.x - 1) ?
    avgChunkSize + (*len % gridDim.x) : avgChunkSize;
  int chunkOffset = 0;

  /* address row split issue. 
   *
   * we do not want a row to be split between
   * blocks so we will edit any necessary chunk sizes such that no
   * elements for the same row are split between two blocks.
   * example: if block n sees that the first e elements in block n+1
   * have the same row as the last elements in n, then n's chunk
   * will be extended by e.
   *
   * similarly, if n notices that the last elements in n-1 have the same row
   * as the first e elements in n, then n's chunk size will be decreased
   * by e and an offset of e will be added to the index of all the elements
   * in the chunk for n.
   */
  int firstElementIndex = (blockDim.x * blockIdx.x);
  int lastElementIndex = (blockDim.x * blockIdx.x) + (chunkSize - 1);

  for(i = firstElementIndex - 1; i >= 0; i--){
    if(r[i] == r[firstElementIndex]){
      chunkSize--;
      chunkOffset++;
    }
    else{
      break;
    }
  }

  for(i = lastElementIndex + 1; i < *len; i++){
    if(r[i] == r[lastElementIndex]){
      chunkSize++;
    }
    else{
      break;
    }
  }

  /* perform the segmented scan. numIterations is how many passes
   * is needed for the segment scan. offset represents how far from
   * the begginning of the chunk we should start performing adds.
   */
  float *tmpNz = nz;
  float *tmpBuf = buffer;
  int numIterations = ceil(log2f((float)chunkSize));
  for(i = 0; i < numIterations; i++){
    nz = (i % 2 == 0) ? tmpNz : tmpBuf;
    buffer = (i % 2 == 0) ? tmpBuf : tmpNz;

    /* this loop handles possible extra elements from a scan
     * if #threads < #elements
     */
    int j;
    int dataIndex = 0;
    for(j = 0; ; j += blockDim.x){
      dataIndex = threadIdx.x + j;
      if(dataIndex >= chunkSize){
        break;
      }

      int offset = (int)powf(2.0, (float)i);
      // index in entire array of nz's
      int globalDataIndex = (avgChunkSize * blockIdx.x) 
        + dataIndex + chunkOffset;

      if(dataIndex >= offset){
        float addend1 = nz[globalDataIndex];
        float addend2 = nz[globalDataIndex - offset];
        int sameRow = r[globalDataIndex] == r[globalDataIndex - offset];

        buffer[globalDataIndex] = (sameRow) ? addend1 + addend2 : addend1;
      }
      else{
        buffer[globalDataIndex] = nz[globalDataIndex];
      }

      // add to output if end of row is reached
      int shouldAdd = (globalDataIndex == (*len - 1)
        || (r[globalDataIndex] != r[globalDataIndex + 1])) 
        && (i == numIterations - 1);
      if(shouldAdd){
        int row = r[globalDataIndex];
        out[row] = buffer[globalDataIndex];
      }
    }

    __syncthreads();
  }
}

int sorter(const void *a, const void *b){
  Nz *typedA = *(Nz**) a;
  Nz *typedB = *(Nz**) b;

  return typedA->r - typedB->r;
}

void getMulScan(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
    /* sort nz's
     * sorting the individual val, rIndex, and cIndex arrays would
     * be annoying so we'll make structs with each of those values
     * and sort that
     */
    Nz **tmpNz = (Nz**) malloc(sizeof(Nz*) * mat->nz);
    int t;
    for(t = 0; t < mat->nz; t++){
      tmpNz[t] = (Nz*) malloc(sizeof(Nz));
      tmpNz[t]->val = mat->val[t];
      tmpNz[t]->r = mat->rIndex[t];
      tmpNz[t]->c = mat->cIndex[t];
    }
    // sort
    qsort(tmpNz, mat->nz, sizeof(Nz*), sorter);
    // adjust mat, cIndex, and rIndex
    for(t=0; t < mat->nz; t++){
      mat->val[t] = tmpNz[t]->val;
      mat->rIndex[t] = tmpNz[t]->r;
      mat->cIndex[t] = tmpNz[t]->c;
    }

    /* map nz's to (nz*vec[properRowIndex]) */
    int i;
    for(i = 0; i < mat->nz; i++){
      mat->val[i] *= vec->val[mat->cIndex[i]];
    }

    /*Allocate things...*/
    // useful sizes
    int intSize = sizeof(int);
    int floatSize = sizeof(float);
    int vecSize = vec->nz;
    int matNzSize = mat->nz;

    int *len;
    int *r;
    float *nz;
    float *out;
    float *buffer;
    cudaMalloc((void**) &len, intSize);
    cudaMalloc((void**) &r, matNzSize*intSize);
    cudaMalloc((void**) &nz, matNzSize*floatSize);
    cudaMalloc((void**) &out, vecSize*floatSize);
    cudaMalloc((void**) &buffer, matNzSize*floatSize);
    cudaMemcpy(len, &matNzSize, intSize, cudaMemcpyHostToDevice);
    cudaMemcpy(r, mat->rIndex, intSize*matNzSize, cudaMemcpyHostToDevice);
    cudaMemcpy(nz, mat->val, floatSize*matNzSize, cudaMemcpyHostToDevice);

    /* record the kernel runtime */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    putProduct<<<blockNum, blockSize>>>(len, r, nz, out, buffer);

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Segmented Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

    /* copy answer into res vector */
    cudaMemcpy(res->val, out, vec->nz*floatSize, cudaMemcpyDeviceToHost);

    /* map nz's back to original form */
    for(i = 0; i < matNzSize; i++){
      mat->val[i] /= vec->val[mat->cIndex[i]];
    }

    /*Deallocate, please*/
    for(t = 0; t < mat->nz; t++){
      free(tmpNz[t]);
    }
    free(tmpNz);
    cudaFree(len);
    cudaFree(r);
    cudaFree(nz);
    cudaFree(out);
    cudaFree(buffer);
}
