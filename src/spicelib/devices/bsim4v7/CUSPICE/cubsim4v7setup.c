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
#include "bsim4v7def.h"
#include "ngspice/CUSPICE/CUSPICE.h"

/* cudaMalloc MACRO to check it for errors --> CUDAMALLOCCHECK(name of pointer, dimension, type, status) */
#define CUDAMALLOCCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuBSIM4v7setup routine...\n") ; \
        fprintf (stderr, "Error: cudaMalloc failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

/* cudaMemcpy MACRO to check it for errors --> CUDAMEMCPYCHECK(name of pointer, dimension, type, status) */
#define CUDAMEMCPYCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuBSIM4v7setup routine...\n") ; \
        fprintf (stderr, "Error: cudaMemcpy failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

int
cuBSIM4v7setup
(
GENmodel *inModel
)
{
    long unsigned int size ;
    cudaError_t status ;
    BSIM4v7model *model = (BSIM4v7model *)inModel ;

    size = (long unsigned int) model->gen.GENnInstances;

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
    model->BSIM4v7paramCPU.BSIM4v7gbsRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gbsRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gbsRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cbsRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cbsRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cbsRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gbdRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gbdRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gbdRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cbdRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cbdRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cbdRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7vonRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7vonRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vonRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7vdsatRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7vdsatRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vdsatRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7csubRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7csubRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7csubRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gdsRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gdsRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gdsRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gmRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gmRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gmRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gmbsRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gmbsRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gmbsRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gcrgRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gcrgRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gcrgRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7IgidlRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7IgidlRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IgidlRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7IgislRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7IgislRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IgislRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7IgcsRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7IgcsRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IgcsRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7IgcdRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7IgcdRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IgcdRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7IgsRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7IgsRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IgsRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7IgdRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7IgdRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IgdRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7IgbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7IgbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IgbRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cdRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cdRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cdRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7qinvRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7qinvRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7qinvRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cggbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cggbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cggbRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cgsbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cgsbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cgsbRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cgdbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cgdbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cgdbRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cdgbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cdgbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cdgbRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cdsbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cdsbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cdsbRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cddbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cddbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cddbRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cbgbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cbgbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cbgbRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cbsbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cbsbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cbsbRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cbdbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cbdbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cbdbRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7csgbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7csgbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7csgbRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cssbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cssbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cssbRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7csdbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7csdbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7csdbRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cgbbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cgbbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cgbbRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7csbbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7csbbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7csbbRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cdbbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cdbbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cdbbRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cbbbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cbbbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cbbbRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gtauRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gtauRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gtauRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7qgateRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7qgateRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7qgateRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7qbulkRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7qbulkRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7qbulkRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7qdrnRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7qdrnRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7qdrnRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7qsrcRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7qsrcRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7qsrcRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7capbsRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7capbsRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7capbsRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7capbdRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7capbdRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7capbdRWArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7icVDSArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7icVDSArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7icVDSArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7icVGSArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7icVGSArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7icVGSArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7icVBSArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7icVBSArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7icVBSArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7vth0Array = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7vth0Array), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vth0Array, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gbbsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gbbsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gbbsArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7ggidlbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7ggidlbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7ggidlbArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gbgsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gbgsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gbgsArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7ggidlgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7ggidlgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7ggidlgArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gbdsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gbdsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gbdsArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7ggidldArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7ggidldArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7ggidldArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7ggislsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7ggislsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7ggislsArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7ggislgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7ggislgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7ggislgArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7ggislbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7ggislbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7ggislbArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gIgsgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gIgsgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgsgArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gIgcsgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gIgcsgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgcsgArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gIgcsdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gIgcsdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgcsdArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gIgcsbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gIgcsbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgcsbArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gIgdgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gIgdgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgdgArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gIgcdgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gIgcdgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgcdgArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gIgcddArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gIgcddArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgcddArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gIgcdbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gIgcdbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgcdbArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gIgbgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gIgbgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgbgArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gIgbdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gIgbdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgbdArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gIgbbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gIgbbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgbbArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7ggidlsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7ggidlsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7ggidlsArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7ggisldArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7ggisldArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7ggisldArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gstotArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gstotArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gstotArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gstotdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gstotdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gstotdArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gstotgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gstotgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gstotgArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gstotbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gstotbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gstotbArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gdtotArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gdtotArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gdtotArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gdtotdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gdtotdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gdtotdArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gdtotgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gdtotgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gdtotgArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gdtotbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gdtotbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gdtotbArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cgdoArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cgdoArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cgdoArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7qgdoArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7qgdoArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7qgdoArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cgsoArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cgsoArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cgsoArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7qgsoArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7qgsoArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7qgsoArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7AseffArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7AseffArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7AseffArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7PseffArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7PseffArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7PseffArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7nfArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7nfArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7nfArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7XExpBVSArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7XExpBVSArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7XExpBVSArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7vjsmFwdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7vjsmFwdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vjsmFwdArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7IVjsmFwdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7IVjsmFwdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IVjsmFwdArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7vjsmRevArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7vjsmRevArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vjsmRevArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7IVjsmRevArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7IVjsmRevArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IVjsmRevArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SslpRevArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SslpRevArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SslpRevArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SslpFwdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SslpFwdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SslpFwdArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7AdeffArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7AdeffArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7AdeffArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7PdeffArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7PdeffArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7PdeffArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7XExpBVDArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7XExpBVDArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7XExpBVDArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7vjdmFwdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7vjdmFwdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vjdmFwdArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7IVjdmFwdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7IVjdmFwdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IVjdmFwdArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7vjdmRevArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7vjdmRevArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vjdmRevArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7IVjdmRevArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7IVjdmRevArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IVjdmRevArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DslpRevArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DslpRevArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DslpRevArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DslpFwdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DslpFwdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DslpFwdArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SjctTempRevSatCurArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SjctTempRevSatCurArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SjctTempRevSatCurArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SswTempRevSatCurArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SswTempRevSatCurArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SswTempRevSatCurArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SswgTempRevSatCurArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SswgTempRevSatCurArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SswgTempRevSatCurArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DjctTempRevSatCurArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DjctTempRevSatCurArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DjctTempRevSatCurArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DswTempRevSatCurArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DswTempRevSatCurArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DswTempRevSatCurArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DswgTempRevSatCurArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DswgTempRevSatCurArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DswgTempRevSatCurArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7vbscArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7vbscArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vbscArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7thetavthArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7thetavthArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7thetavthArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7eta0Array = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7eta0Array), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7eta0Array, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7k2oxArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7k2oxArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7k2oxArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7nstarArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7nstarArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7nstarArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7vfbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7vfbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vfbArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7vgs_effArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7vgs_effArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vgs_effArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7vgd_effArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7vgd_effArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vgd_effArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7dvgs_eff_dvgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7dvgs_eff_dvgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7dvgs_eff_dvgArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7dvgd_eff_dvgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7dvgd_eff_dvgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7dvgd_eff_dvgArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7VgsteffArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7VgsteffArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7VgsteffArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7grdswArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7grdswArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7grdswArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7AbulkArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7AbulkArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7AbulkArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7vtfbphi1Array = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7vtfbphi1Array), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vtfbphi1Array, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7ueffArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7ueffArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7ueffArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7u0tempArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7u0tempArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7u0tempArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7vsattempArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7vsattempArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vsattempArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7EsatLArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7EsatLArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7EsatLArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7VdseffArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7VdseffArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7VdseffArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7vtfbphi2Array = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7vtfbphi2Array), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vtfbphi2Array, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7CoxeffArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7CoxeffArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7CoxeffArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7AbovVgst2VtmArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7AbovVgst2VtmArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7AbovVgst2VtmArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7IdovVdsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7IdovVdsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IdovVdsArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gcrgdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gcrgdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gcrgdArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gcrgbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gcrgbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gcrgbArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gcrggArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gcrggArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gcrggArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7grgeltdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7grgeltdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7grgeltdArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gcrgsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gcrgsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gcrgsArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7sourceConductanceArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7sourceConductanceArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7sourceConductanceArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7drainConductanceArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7drainConductanceArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7drainConductanceArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gstotsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gstotsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gstotsArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gdtotsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gdtotsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gdtotsArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7vfbzbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7vfbzbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vfbzbArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gIgssArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gIgssArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgssArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gIgddArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gIgddArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgddArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gIgbsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gIgbsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgbsArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gIgcssArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gIgcssArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgcssArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gIgcdsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gIgcdsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgcdsArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7noiGd0Array = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7noiGd0Array), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7noiGd0Array, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cqdbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cqdbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cqdbArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cqsbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cqsbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cqsbArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cqgbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cqgbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cqgbArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7qchqsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7qchqsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7qchqsArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7cqbbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7cqbbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cqbbArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7taunetArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7taunetArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7taunetArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gtgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gtgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gtgArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gtdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gtdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gtdArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gtsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gtsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gtsArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gtbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gtbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gtbArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7mArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7mArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7mArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7grbpdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7grbpdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7grbpdArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7grbdbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7grbdbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7grbdbArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7grbpbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7grbpbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7grbpbArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7grbpsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7grbpsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7grbpsArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7grbsbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7grbsbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7grbsbArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7dNodePrimeRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7dNodePrimeRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7dNodePrimeRHSValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gNodePrimeRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gNodePrimeRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gNodePrimeRHSValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gNodeExtRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gNodeExtRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gNodeExtRHSValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7gNodeMidRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gNodeMidRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gNodeMidRHSValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7bNodePrimeRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7bNodePrimeRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7bNodePrimeRHSValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7sNodePrimeRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7sNodePrimeRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7sNodePrimeRHSValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7dbNodeRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7dbNodeRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7dbNodeRHSValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7sbNodeRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7sbNodeRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7sbNodeRHSValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7dNodeRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7dNodeRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7dNodeRHSValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7sNodeRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7sNodeRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7sNodeRHSValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7qNodeRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7qNodeRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7qNodeRHSValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7GEgeValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7GEgeValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7GEgeValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7GPgeValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7GPgeValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7GPgeValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7GEgpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7GEgpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7GEgpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7GPgpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7GPgpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7GPgpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7GPdpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7GPdpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7GPdpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7GPspValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7GPspValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7GPspValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7GPbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7GPbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7GPbpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7GEdpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7GEdpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7GEdpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7GEspValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7GEspValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7GEspValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7GEbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7GEbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7GEbpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7GEgmValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7GEgmValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7GEgmValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7GMgeValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7GMgeValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7GMgeValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7GMgmValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7GMgmValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7GMgmValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7GMdpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7GMdpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7GMdpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7GMgpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7GMgpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7GMgpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7GMspValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7GMspValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7GMspValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7GMbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7GMbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7GMbpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DPgmValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DPgmValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DPgmValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7GPgmValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7GPgmValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7GPgmValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SPgmValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SPgmValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SPgmValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7BPgmValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7BPgmValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7BPgmValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DgpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DgpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DgpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DspValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DspValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DspValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DbpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SdpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SdpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SdpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SgpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SgpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SgpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SbpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DPdpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DPdpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DPdpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DPdValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DPdValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DPdValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DPgpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DPgpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DPgpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DPspValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DPspValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DPspValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DPbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DPbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DPbpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DdpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DdpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DdpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DdValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DdValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DdValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SPdpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SPdpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SPdpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SPgpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SPgpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SPgpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SPspValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SPspValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SPspValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SPsValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SPsValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SPsValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SPbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SPbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SPbpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SspValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SspValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SspValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SsValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SsValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SsValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7BPdpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7BPdpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7BPdpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7BPgpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7BPgpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7BPgpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7BPspValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7BPspValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7BPspValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7BPbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7BPbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7BPbpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DPdbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DPdbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DPdbValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SPsbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SPsbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SPsbValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DBdpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DBdpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DBdpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DBdbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DBdbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DBdbValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DBbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DBbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DBbpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DBbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DBbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DBbValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7BPdbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7BPdbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7BPdbValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7BPbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7BPbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7BPbValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7BPsbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7BPsbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7BPsbValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7BPbpIFValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7BPbpIFValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7BPbpIFValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SBspValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SBspValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SBspValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SBbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SBbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SBbpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SBbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SBbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SBbValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SBsbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SBsbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SBsbValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7BdbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7BdbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7BdbValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7BbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7BbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7BbpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7BsbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7BsbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7BsbValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7BbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7BbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7BbValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7QqValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7QqValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7QqValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7QgpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7QgpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7QgpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7QdpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7QdpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7QdpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7QspValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7QspValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7QspValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7QbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7QbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7QbpValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7DPqValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7DPqValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DPqValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7SPqValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7SPqValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SPqValueArray, size, double, status)

    model->BSIM4v7paramCPU.BSIM4v7GPqValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7GPqValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7GPqValueArray, size, double, status)

    /* INT */
    model->BSIM4v7paramCPU.BSIM4v7offArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7offArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7offArray, size, int, status)

    model->BSIM4v7paramCPU.BSIM4v7dNodePrimeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7dNodePrimeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7dNodePrimeArray, size, int, status)

    model->BSIM4v7paramCPU.BSIM4v7sNodePrimeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7sNodePrimeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7sNodePrimeArray, size, int, status)

    model->BSIM4v7paramCPU.BSIM4v7gNodePrimeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gNodePrimeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gNodePrimeArray, size, int, status)

    model->BSIM4v7paramCPU.BSIM4v7bNodePrimeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7bNodePrimeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7bNodePrimeArray, size, int, status)

    model->BSIM4v7paramCPU.BSIM4v7gNodeExtArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gNodeExtArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gNodeExtArray, size, int, status)

    model->BSIM4v7paramCPU.BSIM4v7gNodeMidArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7gNodeMidArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gNodeMidArray, size, int, status)

    model->BSIM4v7paramCPU.BSIM4v7dbNodeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7dbNodeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7dbNodeArray, size, int, status)

    model->BSIM4v7paramCPU.BSIM4v7sbNodeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7sbNodeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7sbNodeArray, size, int, status)

    model->BSIM4v7paramCPU.BSIM4v7sNodeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7sNodeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7sNodeArray, size, int, status)

    model->BSIM4v7paramCPU.BSIM4v7dNodeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7dNodeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7dNodeArray, size, int, status)

    model->BSIM4v7paramCPU.BSIM4v7qNodeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7qNodeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7qNodeArray, size, int, status)

    model->BSIM4v7paramCPU.BSIM4v7rbodyModArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7rbodyModArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7rbodyModArray, size, int, status)

    model->BSIM4v7paramCPU.BSIM4v7modeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7modeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7modeArray, size, int, status)

    model->BSIM4v7paramCPU.BSIM4v7rgateModArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7rgateModArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7rgateModArray, size, int, status)

    model->BSIM4v7paramCPU.BSIM4v7trnqsModArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7trnqsModArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7trnqsModArray, size, int, status)

    model->BSIM4v7paramCPU.BSIM4v7acnqsModArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7acnqsModArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7acnqsModArray, size, int, status)

    model->BSIM4v7paramCPU.BSIM4v7statesArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4v7paramGPU.d_BSIM4v7statesArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4v7paramGPU.d_BSIM4v7statesArray, size, int, status)

    return (OK) ;
}
