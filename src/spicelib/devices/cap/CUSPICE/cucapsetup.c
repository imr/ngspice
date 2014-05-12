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
#include "capdefs.h"
#include "ngspice/CUSPICE/CUSPICE.h"

/* cudaMalloc MACRO to check it for errors --> CUDAMALLOCCHECK(name of pointer, dimension, type, status) */
#define CUDAMALLOCCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuCAPsetup routine...\n") ; \
        fprintf (stderr, "Error: cudaMalloc failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

/* cudaMemcpy MACRO to check it for errors --> CUDAMEMCPYCHECK(name of pointer, dimension, type, status) */
#define CUDAMEMCPYCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuCAPsetup routine...\n") ; \
        fprintf (stderr, "Error: cudaMemcpy failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

int
cuCAPsetup
(
GENmodel *inModel
)
{
    long unsigned int size ;
    cudaError_t status ;
    CAPmodel *model = (CAPmodel *)inModel ;

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

    status = cudaMalloc ((void **)&(model->d_PositionVector_timeSteps), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->d_PositionVector_timeSteps, size, int, status)

    status = cudaMemcpy (model->d_PositionVector_timeSteps, model->PositionVector_timeSteps, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->d_PositionVector_timeSteps, size, int, status)

    /* DOUBLE */
    model->CAPparamCPU.CAPinitCondArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->CAPparamGPU.d_CAPinitCondArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->CAPparamGPU.d_CAPinitCondArray, size, double, status)

    model->CAPparamCPU.CAPcapacArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->CAPparamGPU.d_CAPcapacArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->CAPparamGPU.d_CAPcapacArray, size, double, status)

    model->CAPparamCPU.CAPmArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->CAPparamGPU.d_CAPmArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->CAPparamGPU.d_CAPmArray, size, double, status)

    model->CAPparamCPU.CAPgeqValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->CAPparamGPU.d_CAPgeqValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->CAPparamGPU.d_CAPgeqValueArray, size, double, status)

    model->CAPparamCPU.CAPceqValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->CAPparamGPU.d_CAPceqValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->CAPparamGPU.d_CAPceqValueArray, size, double, status)

    /* INT */
    model->CAPparamCPU.CAPposNodeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->CAPparamGPU.d_CAPposNodeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->CAPparamGPU.d_CAPposNodeArray, size, int, status)

    model->CAPparamCPU.CAPnegNodeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->CAPparamGPU.d_CAPnegNodeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->CAPparamGPU.d_CAPnegNodeArray, size, int, status)

    model->CAPparamCPU.CAPstateArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->CAPparamGPU.d_CAPstateArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->CAPparamGPU.d_CAPstateArray, size, int, status)

    return (OK) ;
}
