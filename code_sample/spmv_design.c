#include "genresult.cuh"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include "spmv.cuh"

void getMulDesign(MatrixInfo * mat, MatrixInfo * vec, MatrixInfo * res, int blockSize, int blockNum){
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

    /* calculate avg row sparsity */
    float avg = 0;
    float sum = 0;
    int i;
    int cur = -1;
    float numItems = 0;
    for(i = 0; i < mat->nz; i++){
      if(cur == -1){
        cur = mat->rIndex[i];
        numItems++;
      }
      else if(cur == mat->rIndex[i]){
        numItems++;
      }
      else{
        sum += (float)numItems / mat->M;
        cur = mat->rIndex[i];
        numItems = 1;
      }
    }
    // the remaining one
    sum += (float)numItems / (float)mat->N;
    // get avg
    avg = sum / (float)mat->M;

    /* record the kernel runtime */
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    // with sparser rows on avg, there will be less wait time
    // for the threads as the atomic operations are executed
    if(avg > .5){
			getMulAtomic(mat, vec, res, blockSize, blockNum);
    }
    else{
			getMulScan(mat, vec, res, blockSize, blockNum);
    }

    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    printf("Segmented Kernel Time: %lu micro-seconds\n", 1000000 * (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1000);

    /*Deallocate, please*/
    for(t = 0; t < mat->nz; t++){
      free(tmpNz[t]);
    }
    free(tmpNz);
}
