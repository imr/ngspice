/**********
Copyright 2014 - NGSPICE Software
Author: 2014 Francesco Lannutti
**********/

#include "ngspice/config.h"
#include "CUSPICE/cucktterr.cuh"
#include "capdefs.h"

extern "C"
__global__ void cuCAPtrunc_kernel (CAPparamGPUstruct, int, double **, double *, double,
                                   int, int, double, double, double, double, double *, int *) ;

extern "C"
int
cuCAPtrunc
(
GENmodel *inModel, CKTcircuit *ckt, double *timeStep
)
{
    (void)timeStep ;

    CAPmodel *model = (CAPmodel *)inModel ;
    int thread_x, thread_y, block_x ;

    cudaError_t status ;

    /*  loop through all the capacitor models */
    for ( ; model != NULL ; model = CAPnextModel(model))
    {
        /* Determining how many blocks should exist in the kernel */
        thread_x = 1 ;
        thread_y = 256 ;
        if (model->n_instances % thread_y != 0)
            block_x = (int)((model->n_instances + thread_y - 1) / thread_y) ;
        else
            block_x = model->n_instances / thread_y ;

        dim3 thread (thread_x, thread_y) ;

        /* Kernel launch */
        status = cudaGetLastError () ; // clear error status

        cuCAPtrunc_kernel <<< block_x, thread >>> (model->CAPparamGPU, model->n_instances,
                                                   ckt->dD_CKTstates, ckt->d_CKTdeltaOld,
                                                   ckt->CKTdelta, ckt->CKTorder, ckt->CKTintegrateMethod,
                                                   ckt->CKTabstol, ckt->CKTreltol, ckt->CKTchgtol, ckt->CKTtrtol,
                                                   ckt->d_CKTtimeSteps, model->d_PositionVector_timeSteps) ;

        cudaDeviceSynchronize () ;

        status = cudaGetLastError () ; // check for launch error
        if (status != cudaSuccess)
        {
            fprintf (stderr, "Kernel launch failure in the Trunc Capacitor Model\n\n") ;
            return (E_NOMEM) ;
        }
    }

    return (OK) ;
}

extern "C"
__global__
void
cuCAPtrunc_kernel
(
CAPparamGPUstruct CAPentry, int n_instances, double **CKTstates,
double *CKTdeltaOld, double CKTdelta, int CKTorder, int CKTintegrateMethod,
double CKTabsTol, double CKTrelTol, double CKTchgTol, double CKTtrTol,
double *CKTtimeSteps, int *PositionVector_timeSteps
)
{
    int instance_ID ;

    instance_ID = threadIdx.y + blockDim.y * blockIdx.x ;
    if (instance_ID < n_instances)
    {
        if (threadIdx.x == 0)
        {
            cuCKTterr (CAPentry.d_CAPstateArray [instance_ID], CKTstates,
                       CKTdeltaOld, CKTdelta, CKTorder, CKTintegrateMethod,
                       CKTabsTol, CKTrelTol, CKTchgTol, CKTtrTol,
                       &(CKTtimeSteps [PositionVector_timeSteps [instance_ID]])) ;
        }
    }

    return ;
}
