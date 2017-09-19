/**********
Copyright 2014 - NGSPICE Software
Author: 2014 Francesco Lannutti
**********/

#include "ngspice/config.h"
#include "ngspice/cktdefs.h"
#include "cuda_runtime_api.h"
#include "ngspice/macros.h"

/* cudaMemcpy MACRO to check it for errors --> CUDAMEMCPYCHECK(name of pointer, dimension, type, status) */
#define CUDAMEMCPYCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuCKTtrunc routine...\n") ; \
        fprintf (stderr, "Error: cudaMemcpy failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

extern "C"
__global__ void cuCKTtrunc_kernel
(
double *, double *, int
) ;

extern "C"
int
cuCKTtrunc
(
CKTcircuit *ckt, double timetemp, double *timeStep
)
{
        long unsigned int size ;
        double timetempGPU ;
        int thread_x, thread_y, block_x ;

        cudaError_t status ;

        /* Determining how many blocks should exist in the kernel */
        thread_x = 1 ;
        thread_y = 256 ;
        if (ckt->total_n_timeSteps % thread_y != 0)
            block_x = (int)((ckt->total_n_timeSteps + thread_y - 1) / thread_y) ;
        else
            block_x = ckt->total_n_timeSteps / thread_y ;

        dim3 thread (thread_x, thread_y) ;

        /* Kernel launch */
        status = cudaGetLastError () ; // clear error status

        cuCKTtrunc_kernel <<< block_x, thread, thread_y * sizeof(double) >>> (ckt->d_CKTtimeSteps, ckt->d_CKTtimeStepsOut, ckt->total_n_timeSteps) ;

        cudaDeviceSynchronize () ;

        status = cudaGetLastError () ; // check for launch error
        if (status != cudaSuccess)
        {
            fprintf (stderr, "Kernel 1 launch failure in cuCKTtrunc\n\n") ;
            return (E_NOMEM) ;
        }

        cuCKTtrunc_kernel <<< 1, thread, thread_y * sizeof(double) >>> (ckt->d_CKTtimeStepsOut, ckt->d_CKTtimeSteps, block_x) ;

        cudaDeviceSynchronize () ;

        status = cudaGetLastError () ; // check for launch error
        if (status != cudaSuccess)
        {
            fprintf (stderr, "Kernel 2 launch failure in cuCKTtrunc\n\n") ;
            return (E_NOMEM) ;
        }

        /* Copy back the reduction result */
        size = (long unsigned int)(1) ;
        status = cudaMemcpy (&timetempGPU, ckt->d_CKTtimeSteps, size * sizeof(double), cudaMemcpyDeviceToHost) ;
        CUDAMEMCPYCHECK (&timetempGPU, size, double, status)

        /* Final Comparison */
        if (timetempGPU < timetemp)
        {
            timetemp = timetempGPU ;
        }
        if (2 * *timeStep < timetemp)
        {
            *timeStep = 2 * *timeStep ;
        } else {
            *timeStep = timetemp ;
        }

    return 0 ;
}

extern "C"
__global__
void
cuCKTtrunc_kernel
(
double *g_idata, double *g_odata, int n
)
{
    extern __shared__ double sdata [] ;
    unsigned int i, tid ;

    tid = threadIdx.y ;
//    i = blockIdx.x * (blockDim.y * 2) + tid ;
    i = blockIdx.x * blockDim.y + tid ;
    if (i < n)
    {
//        sdata [tid] = MIN (g_idata [i], g_idata [i + blockDim.y]) ;
        sdata [tid] = g_idata [i] ;
    }
    __syncthreads () ;

    if ((tid < 128) && (i + 128 < n))
    {
        sdata [tid] = MIN (sdata [tid], sdata [tid + 128]) ;
    }
    __syncthreads () ;

    if ((tid < 64) && (i + 64 < n))
    {
        sdata [tid] = MIN (sdata [tid], sdata [tid + 64]) ;
    }
    __syncthreads () ;

    if ((tid < 32) && (i + 32 < n))
    {
        sdata [tid] = MIN (sdata [tid], sdata [tid + 32]) ;
        sdata [tid] = MIN (sdata [tid], sdata [tid + 16]) ;
        sdata [tid] = MIN (sdata [tid], sdata [tid + 8]) ;
        sdata [tid] = MIN (sdata [tid], sdata [tid + 4]) ;
        sdata [tid] = MIN (sdata [tid], sdata [tid + 2]) ;
        sdata [tid] = MIN (sdata [tid], sdata [tid + 1]) ;
    }

    if (tid == 0)
    {
        g_odata [blockIdx.x] = sdata [0] ;
    }
}
