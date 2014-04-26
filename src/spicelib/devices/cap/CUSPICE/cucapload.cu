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
#include "capdefs.h"

extern "C"
__global__ void cuCAPload_kernel (CAPparamGPUstruct, double *, double *, double *,
                                  int, double, double, int, int, int, int *, double *, int *, double *) ;

extern "C"
int
cuCAPload
(
GENmodel *inModel, CKTcircuit *ckt
)
{
    CAPmodel *model = (CAPmodel *)inModel ;
    int cond1, thread_x, thread_y, block_x ;

    cudaError_t status ;

    /* check if capacitors are in the circuit or are open circuited */
    if (ckt->CKTmode & (MODETRAN|MODEAC|MODETRANOP))
    {
        /* evaluate device independent analysis conditions */
        cond1 = (((ckt->CKTmode & MODEDC) && (ckt->CKTmode & MODEINITJCT))
                || ((ckt->CKTmode & MODEUIC) && (ckt->CKTmode & MODEINITTRAN))) ;

        /*  loop through all the resistor models */
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

            cuCAPload_kernel <<< block_x, thread >>> (model->CAPparamGPU, ckt->d_CKTrhsOld, ckt->d_CKTstate0,
                                                      ckt->d_CKTstate1, ckt->CKTmode, ckt->CKTag [0], ckt->CKTag [1],
                                                      ckt->CKTorder, model->n_instances, cond1,
                                                      model->d_PositionVector, ckt->d_CKTloadOutput,
                                                      model->d_PositionVectorRHS, ckt->d_CKTloadOutputRHS) ;

            cudaDeviceSynchronize () ;

            status = cudaGetLastError () ; // check for launch error
            if (status != cudaSuccess)
            {
                fprintf (stderr, "Kernel launch failure in the Capacitor Model\n\n") ;
                return (E_NOMEM) ;
            }
        }
    }

    return (OK) ;
}

extern "C"
__global__
void
cuCAPload_kernel
(
CAPparamGPUstruct CAPentry, double *CKTrhsOld, double *CKTstate_0,
double *CKTstate_1, int CKTmode, double CKTag_0, double CKTag_1,
int CKTorder, int n_instances, int cond1, int *d_PositionVector,
double *d_CKTloadOutput, int *d_PositionVectorRHS, double *d_CKTloadOutputRHS
)
{
    int instance_ID ;
    double vcap, geq, ceq, m ;
    int error ;

    instance_ID = threadIdx.y + blockDim.y * blockIdx.x ;

    if (instance_ID < n_instances)
    {
        if (threadIdx.x == 0)
        {
            m = CAPentry.d_CAPmArray [instance_ID] ;

            if (cond1)
            {
                vcap = CAPentry.d_CAPinitCondArray [instance_ID] ;
            } else {
                vcap = CKTrhsOld [CAPentry.d_CAPposNodeArray [instance_ID]] -
                       CKTrhsOld [CAPentry.d_CAPnegNodeArray [instance_ID]] ;
            }

            if (CKTmode & (MODETRAN | MODEAC))
            {
#ifndef PREDICTOR
                if (CKTmode & MODEINITPRED)
                {
                    CKTstate_0 [CAPentry.d_CAPstateArray [instance_ID]] =
                    CKTstate_1 [CAPentry.d_CAPstateArray [instance_ID]] ;
                } else { /* only const caps - no poly's */
#endif /* PREDICTOR */
                    CKTstate_0 [CAPentry.d_CAPstateArray [instance_ID]] = CAPentry.d_CAPcapacArray [instance_ID] * vcap ;
                    if (CKTmode & MODEINITTRAN)
                    {
                        CKTstate_1 [CAPentry.d_CAPstateArray [instance_ID]] =
                        CKTstate_0 [CAPentry.d_CAPstateArray [instance_ID]] ;
                    }
#ifndef PREDICTOR
                }
#endif /* PREDICTOR */
                error = cuNIintegrate_device_kernel (CKTstate_0, CKTstate_1, &geq, &ceq,
                                                    CAPentry.d_CAPcapacArray [instance_ID],
                                                    CAPentry.d_CAPstateArray [instance_ID],
                                                    CKTag_0, CKTag_1, CKTorder) ;
                if (error)
                    printf ("Error in the integration!\n\n") ;
                    //return (error) ;

                if (CKTmode & MODEINITTRAN)
                {
                    CKTstate_1 [CAPentry.d_CAPstateArray [instance_ID] + 1] =
                    CKTstate_0 [CAPentry.d_CAPstateArray [instance_ID] + 1] ;
                }

                d_CKTloadOutput [d_PositionVector [instance_ID]] = m * geq ;
                d_CKTloadOutputRHS [d_PositionVectorRHS [instance_ID]] = m * ceq ;

            } else {
                CKTstate_0 [CAPentry.d_CAPstateArray [instance_ID]] = CAPentry.d_CAPcapacArray [instance_ID] * vcap ;
            }
        }
    }

    return ;
}
