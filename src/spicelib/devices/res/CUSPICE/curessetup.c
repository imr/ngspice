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
#include "cuda_runtime_api.h"
#include "resdefs.h"
#include "ngspice/CUSPICE/CUSPICE.h"

/* cudaMalloc MACRO to check it for errors --> CUDAMALLOCCHECK(name of pointer, dimension, type, status) */
#define CUDAMALLOCCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuRESsetup routine...\n") ; \
        fprintf (stderr, "Error: cudaMalloc failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

/* cudaMemcpy MACRO to check it for errors --> CUDAMEMCPYCHECK(name of pointer, dimension, type, status) */
#define CUDAMEMCPYCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuRESsetup routine...\n") ; \
        fprintf (stderr, "Error: cudaMemcpy failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

int
cuRESsetup
(
GENmodel *inModel
)
{
    long unsigned int size ;
    cudaError_t status ;
    RESmodel *model = (RESmodel *)inModel ;

    size = (long unsigned int) model->n_instances ;

    /* Space Allocation to GPU */
    status = cudaMalloc ((void **)&(model->d_PositionVector), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->d_PositionVector, size, int, status)

    status = cudaMemcpy (model->d_PositionVector, model->PositionVector, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->d_PositionVector, size, int, status)

    /* DOUBLE */
    model->RESparamCPU.RESconductArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->RESparamGPU.d_RESconductArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->RESparamGPU.d_RESconductArray, size, double, status)

    model->RESparamCPU.REScurrentArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->RESparamGPU.d_REScurrentArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->RESparamGPU.d_REScurrentArray, size, double, status)

    /* INT */
    model->RESparamCPU.RESposNodeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->RESparamGPU.d_RESposNodeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->RESparamGPU.d_RESposNodeArray, size, int, status)

    model->RESparamCPU.RESnegNodeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->RESparamGPU.d_RESnegNodeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->RESparamGPU.d_RESnegNodeArray, size, int, status)

    return (OK) ;
}
