/*
 * Copyright (c) 2014, NVIDIA Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, 
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, 
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, 
 *    this list of conditions and the following disclaimer in the documentation and/or 
 *    other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to 
 *    endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, 
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ngspice/config.h"
#include "ngspice/CUSPICE/cuniinteg.cuh"
#include "inddefs.h"

/* cudaMalloc MACRO to check it for errors --> CUDAMALLOCCHECK(name of pointer, dimension, type, status) */
#define CUDAMALLOCCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuINDload routine...\n") ; \
        fprintf (stderr, "Error: cudaMalloc failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

/* cudaMemcpy MACRO to check it for errors --> CUDAMEMCPYCHECK(name of pointer, dimension, type, status) */
#define CUDAMEMCPYCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuINDload routine...\n") ; \
        fprintf (stderr, "Error: cudaMemcpy failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

extern "C"
__global__ void cuINDload_kernel (INDparamGPUstruct, double *, double *, double *, int, double, double, int, int, int *, double *, int *, double *) ;

extern "C"
int
cuINDload
(
GENmodel *inModel, CKTcircuit *ckt
)
{
    INDmodel *model = (INDmodel *)inModel ;
    int thread_x, thread_y, block_x ;

    cudaError_t status ;
    
    /*  loop through all the inductor models */
    for ( ; model != NULL ; model = INDnextModel(model))
    {
        /* Determining how many blocks should exist in the kernel */
        thread_x = 1 ;
        thread_y = 256 ;
        if (model->n_instances % thread_y != 0)
            block_x = (int)(model->n_instances / thread_y) + 1 ;
        else
            block_x = model->n_instances / thread_y ;

        dim3 thread (thread_x, thread_y) ;

        /* Kernel launch */
        status = cudaGetLastError () ; // clear error status

        cuINDload_kernel <<< block_x, thread >>> (model->INDparamGPU, ckt->d_CKTrhsOld, ckt->d_CKTstate0,
                                                  ckt->d_CKTstate1, ckt->CKTmode, ckt->CKTag [0], ckt->CKTag [1],
                                                  ckt->CKTorder, model->n_instances,
                                                  model->d_PositionVector, ckt->d_CKTloadOutput,
                                                  model->d_PositionVectorRHS, ckt->d_CKTloadOutputRHS) ;

        cudaDeviceSynchronize () ;

        status = cudaGetLastError () ; // check for launch error
        if (status != cudaSuccess)
        {
            fprintf (stderr, "Kernel launch failure in the Inductor Model\n\n") ;
            return (E_NOMEM) ;
        }
    }

    return (OK) ;
}

extern "C"
__global__
void
cuINDload_kernel
(
INDparamGPUstruct INDentry, double *CKTrhsOld, double *CKTstate_0,
double *CKTstate_1, int CKTmode, double CKTag_0, double CKTag_1,
int CKTorder, int ind_n_instances,
int *d_PositionVector, double *d_CKTloadOutput,
int *d_PositionVectorRHS, double *d_CKTloadOutputRHS
)
{
    int instance_ID ;
    int error ;
    double req, veq ;

    instance_ID = threadIdx.y + blockDim.y * blockIdx.x ;

    if (instance_ID < ind_n_instances)
    {
        if (threadIdx.x == 0)
        {
            if (!(CKTmode & (MODEDC | MODEINITPRED)))
            {
                if (CKTmode & MODEUIC && CKTmode & MODEINITTRAN)
                    CKTstate_0 [INDentry.d_INDstateArray [instance_ID]] =
                    INDentry.d_INDinductArray [instance_ID] * INDentry.d_INDinitCondArray [instance_ID] ;
                else
                    CKTstate_0 [INDentry.d_INDstateArray [instance_ID]] =
                    INDentry.d_INDinductArray [instance_ID] * CKTrhsOld [INDentry.d_INDbrEqArray [instance_ID]] ;
            }

            if (CKTmode & MODEDC)
            {
                req = 0.0 ;
                veq = 0.0 ;
            } else {
#ifndef PREDICTOR
                if (CKTmode & MODEINITPRED)
                    CKTstate_0 [INDentry.d_INDstateArray [instance_ID]] =
                    CKTstate_1 [INDentry.d_INDstateArray [instance_ID]] ;
                else
#endif /*PREDICTOR*/
                    if (CKTmode & MODEINITTRAN)
                        CKTstate_1 [INDentry.d_INDstateArray [instance_ID]] =
                        CKTstate_0 [INDentry.d_INDstateArray [instance_ID]] ;

                error = cuNIintegrate_device_kernel (CKTstate_0, CKTstate_1, &req, &veq,
                                                    INDentry.d_INDinductArray [instance_ID],
                                                    INDentry.d_INDstateArray [instance_ID],
                                                    CKTag_0, CKTag_1, CKTorder) ;
                if (error)
                    printf ("Error in the integration!\n\n") ;
                    //return (error) ;
            }
            
            if (CKTmode & MODEINITTRAN)
                CKTstate_1 [INDentry.d_INDstateArray [instance_ID] + 1] =
                CKTstate_0 [INDentry.d_INDstateArray [instance_ID] + 1] ;

            /* Output for the Matrix */
            d_CKTloadOutput [d_PositionVector [instance_ID]] = 1.0 ;
            d_CKTloadOutput [d_PositionVector [instance_ID] + 1] = req ;

            /* Output for the RHS */
            d_CKTloadOutputRHS [d_PositionVectorRHS [instance_ID]] = veq ;
        }
    }

    return ;
}
