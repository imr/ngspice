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
#include "resdefs.h"

extern "C"
__global__ void cuRESload_kernel (RESparamGPUstruct, double *, int, int *, double *) ;

extern "C"
int
cuRESload
(
GENmodel *inModel, CKTcircuit *ckt
)
{
    RESmodel *model = (RESmodel *)inModel ;
    int thread_x, thread_y, block_x ;

    cudaError_t status ;

    /*  loop through all the resistor models */
    for ( ; model != NULL ; model = RESnextModel(model))
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

        cuRESload_kernel <<< block_x, thread >>> (model->RESparamGPU, ckt->d_CKTrhsOld, model->n_instances,
                                                  model->d_PositionVector, ckt->d_CKTloadOutput) ;

        cudaDeviceSynchronize () ;

        status = cudaGetLastError () ; // check for launch error
        if (status != cudaSuccess)
        {
            fprintf (stderr, "Kernel launch failure in the Resistor Model\n\n") ;
            return (E_NOMEM) ;
        }
    }

    return (OK) ;
}

extern "C"
__global__
void
cuRESload_kernel
(
RESparamGPUstruct RESentry, double *CKTrhsOld, int n_instances, int *d_PositionVector, double * d_CKTloadOutput
)
{
    int instance_ID ;

    instance_ID = threadIdx.y + blockDim.y * blockIdx.x ;
    if (instance_ID < n_instances)
    {
        if (threadIdx.x == 0)
        {
            RESentry.d_REScurrentArray [instance_ID] = (CKTrhsOld [RESentry.d_RESposNodeArray [instance_ID]] -
                                                    CKTrhsOld [RESentry.d_RESnegNodeArray [instance_ID]]) *
                                                    RESentry.d_RESconductArray [instance_ID] ;

            d_CKTloadOutput [d_PositionVector [instance_ID]] = RESentry.d_RESconductArray [instance_ID] ;
        }
    }

    return ;
}
