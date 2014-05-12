/**********
Copyright 2014 - NGSPICE Software
Author: 2014 Francesco Lannutti
**********/

#include "ngspice/config.h"
#include "CUSPICE/cucktterr.cuh"
#include "bsim4v7def.h"

extern "C"
__global__ void cuBSIM4v7trunc_kernel (BSIM4v7paramGPUstruct, int, double **, double *, double,
                                     int, int, double, double, double, double, double *, int *) ;

extern "C"
int
cuBSIM4v7trunc
(
GENmodel *inModel, CKTcircuit *ckt, double *timeStep
)
{
    (void)timeStep ;

    BSIM4v7model *model = (BSIM4v7model *)inModel ;
    int thread_x, thread_y, block_x ;

    cudaError_t status ;

    /*  loop through all the BSIM4v7 models */
    for ( ; model != NULL ; model = BSIM4v7nextModel(model))
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

        cuBSIM4v7trunc_kernel <<< block_x, thread >>> (model->BSIM4v7paramGPU, model->n_instances,
                                                     ckt->dD_CKTstates, ckt->d_CKTdeltaOld,
                                                     ckt->CKTdelta, ckt->CKTorder, ckt->CKTintegrateMethod,
                                                     ckt->CKTabstol, ckt->CKTreltol, ckt->CKTchgtol, ckt->CKTtrtol,
                                                     ckt->d_CKTtimeSteps, model->d_PositionVector_timeSteps) ;

        cudaDeviceSynchronize () ;

        status = cudaGetLastError () ; // check for launch error
        if (status != cudaSuccess)
        {
            fprintf (stderr, "Kernel launch failure in the Trunc BSIM4v7 Model\n\n") ;
            return (E_NOMEM) ;
        }
    }

    return (OK) ;
}

extern "C"
__global__
void
cuBSIM4v7trunc_kernel
(
BSIM4v7paramGPUstruct BSIM4v7entry, int n_instances, double **CKTstates,
double *CKTdeltaOld, double CKTdelta, int CKTorder, int CKTintegrateMethod,
double CKTabsTol, double CKTrelTol, double CKTchgTol, double CKTtrTol,
double *CKTtimeSteps, int *PositionVector_timeSteps
)
{
    int instance_ID, i ;

    instance_ID = threadIdx.y + blockDim.y * blockIdx.x ;
    if (instance_ID < n_instances)
    {
        if (threadIdx.x == 0)
        {
            i = 0 ;

            cuCKTterr (BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 11, CKTstates,
                       CKTdeltaOld, CKTdelta, CKTorder, CKTintegrateMethod,
                       CKTabsTol, CKTrelTol, CKTchgTol, CKTtrTol,
                       &(CKTtimeSteps [PositionVector_timeSteps [instance_ID] + i])) ;
            i++ ;

            cuCKTterr (BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 13, CKTstates,
                       CKTdeltaOld, CKTdelta, CKTorder, CKTintegrateMethod,
                       CKTabsTol, CKTrelTol, CKTchgTol, CKTtrTol,
                       &(CKTtimeSteps [PositionVector_timeSteps [instance_ID] + i])) ;
            i++ ;

            cuCKTterr (BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 15, CKTstates,
                       CKTdeltaOld, CKTdelta, CKTorder, CKTintegrateMethod,
                       CKTabsTol, CKTrelTol, CKTchgTol, CKTtrTol,
                       &(CKTtimeSteps [PositionVector_timeSteps [instance_ID] + i])) ;
            i++ ;

            if (BSIM4v7entry.d_BSIM4v7trnqsModArray [instance_ID])
            {
                cuCKTterr (BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 25, CKTstates,
                           CKTdeltaOld, CKTdelta, CKTorder, CKTintegrateMethod,
                           CKTabsTol, CKTrelTol, CKTchgTol, CKTtrTol,
                           &(CKTtimeSteps [PositionVector_timeSteps [instance_ID] + i])) ;
                i++ ;
            }

            if (BSIM4v7entry.d_BSIM4v7rbodyModArray [instance_ID])
            {
                cuCKTterr (BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 19, CKTstates,
                           CKTdeltaOld, CKTdelta, CKTorder, CKTintegrateMethod,
                           CKTabsTol, CKTrelTol, CKTchgTol, CKTtrTol,
                           &(CKTtimeSteps [PositionVector_timeSteps [instance_ID] + i])) ;
                i++ ;

                cuCKTterr (BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 21, CKTstates,
                           CKTdeltaOld, CKTdelta, CKTorder, CKTintegrateMethod,
                           CKTabsTol, CKTrelTol, CKTchgTol, CKTtrTol,
                           &(CKTtimeSteps [PositionVector_timeSteps [instance_ID] + i])) ;
                i++ ;
            }

            if (BSIM4v7entry.d_BSIM4v7rgateModArray [instance_ID] == 3)
            {
                cuCKTterr (BSIM4v7entry.d_BSIM4v7statesArray [instance_ID] + 17, CKTstates,
                           CKTdeltaOld, CKTdelta, CKTorder, CKTintegrateMethod,
                           CKTabsTol, CKTrelTol, CKTchgTol, CKTtrTol,
                           &(CKTtimeSteps [PositionVector_timeSteps [instance_ID] + i])) ;
                i++ ;
            }
        }
    }

    return ;
}
