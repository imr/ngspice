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
#include "inddefs.h"
#include "ngspice/CUSPICE/CUSPICE.h"

/* cudaMalloc MACRO to check it for errors --> CUDAMALLOCCHECK(name of pointer, dimension, type, status) */
#define CUDAMALLOCCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuINDsetup routine...\n") ; \
        fprintf (stderr, "Error: cudaMalloc failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

/* cudaMemcpy MACRO to check it for errors --> CUDAMEMCPYCHECK(name of pointer, dimension, type, status) */
#define CUDAMEMCPYCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuINDsetup routine...\n") ; \
        fprintf (stderr, "Error: cudaMemcpy failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

int
cuINDsetup
(
GENmodel *inModel
)
{
    long unsigned int size ;
    cudaError_t status ;
    INDmodel *model = (INDmodel *)inModel ;

    size = (long unsigned int) model->n_instances;

    /* Space Allocation to GPU */
    status = cudaMalloc ((void **)&(model->d_PositionVector), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->d_PositionVector, size, int, status)

    status = cudaMemcpy (model->d_PositionVector, model->PositionVector, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->d_PositionVector, size, int, status)

    status = cudaMalloc ((void **)&(model->d_PositionVectorRHS), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->d_PositionVectorRHS, size, int, status)

    status = cudaMemcpy (model->d_PositionVectorRHS, model->PositionVectorRHS, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->d_PositionVectorRHS, size, int, status)

    /* DOUBLE */
    model->INDparamCPU.INDinitCondArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->INDparamGPU.d_INDinitCondArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->INDparamGPU.d_INDinitCondArray, size, double, status)

    model->INDparamCPU.INDinductArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->INDparamGPU.d_INDinductArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->INDparamGPU.d_INDinductArray, size, double, status)

    model->INDparamCPU.INDreqValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->INDparamGPU.d_INDreqValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->INDparamGPU.d_INDreqValueArray, size, double, status)

    model->INDparamCPU.INDveqValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->INDparamGPU.d_INDveqValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->INDparamGPU.d_INDveqValueArray, size, double, status)

    /* INT */
    model->INDparamCPU.INDbrEqArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->INDparamGPU.d_INDbrEqArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->INDparamGPU.d_INDbrEqArray, size, int, status)

    model->INDparamCPU.INDstateArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->INDparamGPU.d_INDstateArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->INDparamGPU.d_INDstateArray, size, int, status)

    return (OK) ;
}
