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
#include "bsim4def.h"
#include "ngspice/CUSPICE/CUSPICE.h"

/* cudaMalloc MACRO to check it for errors --> CUDAMALLOCCHECK(name of pointer, dimension, type, status) */
#define CUDAMALLOCCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuBSIM4setup routine...\n") ; \
        fprintf (stderr, "Error: cudaMalloc failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

/* cudaMemcpy MACRO to check it for errors --> CUDAMEMCPYCHECK(name of pointer, dimension, type, status) */
#define CUDAMEMCPYCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuBSIM4setup routine...\n") ; \
        fprintf (stderr, "Error: cudaMemcpy failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

int
cuBSIM4setup
(
GENmodel *inModel
)
{
    long unsigned int size ;
    cudaError_t status ;
    BSIM4model *model = (BSIM4model *)inModel ;

    size = (long unsigned int)model->n_instances ;

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
    model->BSIM4paramCPU.BSIM4gbsRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gbsRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gbsRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cbsRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cbsRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cbsRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gbdRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gbdRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gbdRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cbdRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cbdRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cbdRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4vonRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4vonRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4vonRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4vdsatRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4vdsatRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4vdsatRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4csubRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4csubRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4csubRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gdsRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gdsRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gdsRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gmRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gmRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gmRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gmbsRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gmbsRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gmbsRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gcrgRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gcrgRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gcrgRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4IgidlRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4IgidlRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4IgidlRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4IgislRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4IgislRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4IgislRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4IgcsRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4IgcsRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4IgcsRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4IgcdRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4IgcdRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4IgcdRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4IgsRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4IgsRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4IgsRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4IgdRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4IgdRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4IgdRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4IgbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4IgbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4IgbRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cdRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cdRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cdRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4qinvRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4qinvRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4qinvRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cggbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cggbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cggbRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cgsbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cgsbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cgsbRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cgdbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cgdbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cgdbRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cdgbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cdgbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cdgbRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cdsbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cdsbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cdsbRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cddbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cddbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cddbRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cbgbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cbgbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cbgbRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cbsbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cbsbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cbsbRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cbdbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cbdbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cbdbRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4csgbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4csgbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4csgbRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cssbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cssbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cssbRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4csdbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4csdbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4csdbRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cgbbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cgbbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cgbbRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4csbbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4csbbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4csbbRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cdbbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cdbbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cdbbRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cbbbRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cbbbRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cbbbRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gtauRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gtauRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gtauRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4qgateRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4qgateRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4qgateRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4qbulkRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4qbulkRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4qbulkRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4qdrnRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4qdrnRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4qdrnRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4qsrcRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4qsrcRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4qsrcRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4capbsRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4capbsRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4capbsRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4capbdRWArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4capbdRWArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4capbdRWArray, size, double, status)

    model->BSIM4paramCPU.BSIM4icVDSArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4icVDSArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4icVDSArray, size, double, status)

    model->BSIM4paramCPU.BSIM4icVGSArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4icVGSArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4icVGSArray, size, double, status)

    model->BSIM4paramCPU.BSIM4icVBSArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4icVBSArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4icVBSArray, size, double, status)

    model->BSIM4paramCPU.BSIM4vth0Array = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4vth0Array), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4vth0Array, size, double, status)

    model->BSIM4paramCPU.BSIM4gbbsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gbbsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gbbsArray, size, double, status)

    model->BSIM4paramCPU.BSIM4ggidlbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4ggidlbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4ggidlbArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gbgsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gbgsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gbgsArray, size, double, status)

    model->BSIM4paramCPU.BSIM4ggidlgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4ggidlgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4ggidlgArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gbdsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gbdsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gbdsArray, size, double, status)

    model->BSIM4paramCPU.BSIM4ggidldArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4ggidldArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4ggidldArray, size, double, status)

    model->BSIM4paramCPU.BSIM4ggislsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4ggislsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4ggislsArray, size, double, status)

    model->BSIM4paramCPU.BSIM4ggislgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4ggislgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4ggislgArray, size, double, status)

    model->BSIM4paramCPU.BSIM4ggislbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4ggislbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4ggislbArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gIgsgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gIgsgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gIgsgArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gIgcsgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gIgcsgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gIgcsgArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gIgcsdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gIgcsdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gIgcsdArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gIgcsbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gIgcsbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gIgcsbArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gIgdgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gIgdgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gIgdgArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gIgcdgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gIgcdgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gIgcdgArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gIgcddArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gIgcddArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gIgcddArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gIgcdbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gIgcdbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gIgcdbArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gIgbgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gIgbgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gIgbgArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gIgbdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gIgbdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gIgbdArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gIgbbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gIgbbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gIgbbArray, size, double, status)

    model->BSIM4paramCPU.BSIM4ggidlsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4ggidlsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4ggidlsArray, size, double, status)

    model->BSIM4paramCPU.BSIM4ggisldArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4ggisldArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4ggisldArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gstotArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gstotArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gstotArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gstotdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gstotdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gstotdArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gstotgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gstotgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gstotgArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gstotbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gstotbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gstotbArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gdtotArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gdtotArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gdtotArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gdtotdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gdtotdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gdtotdArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gdtotgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gdtotgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gdtotgArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gdtotbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gdtotbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gdtotbArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cgdoArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cgdoArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cgdoArray, size, double, status)

    model->BSIM4paramCPU.BSIM4qgdoArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4qgdoArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4qgdoArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cgsoArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cgsoArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cgsoArray, size, double, status)

    model->BSIM4paramCPU.BSIM4qgsoArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4qgsoArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4qgsoArray, size, double, status)

    model->BSIM4paramCPU.BSIM4AseffArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4AseffArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4AseffArray, size, double, status)

    model->BSIM4paramCPU.BSIM4PseffArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4PseffArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4PseffArray, size, double, status)

    model->BSIM4paramCPU.BSIM4nfArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4nfArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4nfArray, size, double, status)

    model->BSIM4paramCPU.BSIM4XExpBVSArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4XExpBVSArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4XExpBVSArray, size, double, status)

    model->BSIM4paramCPU.BSIM4vjsmFwdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4vjsmFwdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4vjsmFwdArray, size, double, status)

    model->BSIM4paramCPU.BSIM4IVjsmFwdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4IVjsmFwdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4IVjsmFwdArray, size, double, status)

    model->BSIM4paramCPU.BSIM4vjsmRevArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4vjsmRevArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4vjsmRevArray, size, double, status)

    model->BSIM4paramCPU.BSIM4IVjsmRevArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4IVjsmRevArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4IVjsmRevArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SslpRevArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SslpRevArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SslpRevArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SslpFwdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SslpFwdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SslpFwdArray, size, double, status)

    model->BSIM4paramCPU.BSIM4AdeffArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4AdeffArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4AdeffArray, size, double, status)

    model->BSIM4paramCPU.BSIM4PdeffArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4PdeffArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4PdeffArray, size, double, status)

    model->BSIM4paramCPU.BSIM4XExpBVDArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4XExpBVDArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4XExpBVDArray, size, double, status)

    model->BSIM4paramCPU.BSIM4vjdmFwdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4vjdmFwdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4vjdmFwdArray, size, double, status)

    model->BSIM4paramCPU.BSIM4IVjdmFwdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4IVjdmFwdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4IVjdmFwdArray, size, double, status)

    model->BSIM4paramCPU.BSIM4vjdmRevArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4vjdmRevArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4vjdmRevArray, size, double, status)

    model->BSIM4paramCPU.BSIM4IVjdmRevArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4IVjdmRevArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4IVjdmRevArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DslpRevArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DslpRevArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DslpRevArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DslpFwdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DslpFwdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DslpFwdArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SjctTempRevSatCurArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SjctTempRevSatCurArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SjctTempRevSatCurArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SswTempRevSatCurArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SswTempRevSatCurArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SswTempRevSatCurArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SswgTempRevSatCurArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SswgTempRevSatCurArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SswgTempRevSatCurArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DjctTempRevSatCurArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DjctTempRevSatCurArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DjctTempRevSatCurArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DswTempRevSatCurArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DswTempRevSatCurArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DswTempRevSatCurArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DswgTempRevSatCurArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DswgTempRevSatCurArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DswgTempRevSatCurArray, size, double, status)

    model->BSIM4paramCPU.BSIM4vbscArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4vbscArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4vbscArray, size, double, status)

    model->BSIM4paramCPU.BSIM4thetavthArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4thetavthArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4thetavthArray, size, double, status)

    model->BSIM4paramCPU.BSIM4eta0Array = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4eta0Array), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4eta0Array, size, double, status)

    model->BSIM4paramCPU.BSIM4k2oxArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4k2oxArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4k2oxArray, size, double, status)

    model->BSIM4paramCPU.BSIM4nstarArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4nstarArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4nstarArray, size, double, status)

    model->BSIM4paramCPU.BSIM4vfbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4vfbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4vfbArray, size, double, status)

    model->BSIM4paramCPU.BSIM4vgs_effArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4vgs_effArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4vgs_effArray, size, double, status)

    model->BSIM4paramCPU.BSIM4vgd_effArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4vgd_effArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4vgd_effArray, size, double, status)

    model->BSIM4paramCPU.BSIM4dvgs_eff_dvgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4dvgs_eff_dvgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4dvgs_eff_dvgArray, size, double, status)

    model->BSIM4paramCPU.BSIM4dvgd_eff_dvgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4dvgd_eff_dvgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4dvgd_eff_dvgArray, size, double, status)

    model->BSIM4paramCPU.BSIM4VgsteffArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4VgsteffArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4VgsteffArray, size, double, status)

    model->BSIM4paramCPU.BSIM4grdswArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4grdswArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4grdswArray, size, double, status)

    model->BSIM4paramCPU.BSIM4AbulkArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4AbulkArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4AbulkArray, size, double, status)

    model->BSIM4paramCPU.BSIM4vtfbphi1Array = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4vtfbphi1Array), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4vtfbphi1Array, size, double, status)

    model->BSIM4paramCPU.BSIM4ueffArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4ueffArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4ueffArray, size, double, status)

    model->BSIM4paramCPU.BSIM4u0tempArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4u0tempArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4u0tempArray, size, double, status)

    model->BSIM4paramCPU.BSIM4vsattempArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4vsattempArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4vsattempArray, size, double, status)

    model->BSIM4paramCPU.BSIM4EsatLArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4EsatLArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4EsatLArray, size, double, status)

    model->BSIM4paramCPU.BSIM4VdseffArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4VdseffArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4VdseffArray, size, double, status)

    model->BSIM4paramCPU.BSIM4vtfbphi2Array = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4vtfbphi2Array), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4vtfbphi2Array, size, double, status)

    model->BSIM4paramCPU.BSIM4CoxeffArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4CoxeffArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4CoxeffArray, size, double, status)

    model->BSIM4paramCPU.BSIM4AbovVgst2VtmArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4AbovVgst2VtmArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4AbovVgst2VtmArray, size, double, status)

    model->BSIM4paramCPU.BSIM4IdovVdsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4IdovVdsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4IdovVdsArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gcrgdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gcrgdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gcrgdArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gcrgbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gcrgbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gcrgbArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gcrggArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gcrggArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gcrggArray, size, double, status)

    model->BSIM4paramCPU.BSIM4grgeltdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4grgeltdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4grgeltdArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gcrgsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gcrgsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gcrgsArray, size, double, status)

    model->BSIM4paramCPU.BSIM4sourceConductanceArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4sourceConductanceArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4sourceConductanceArray, size, double, status)

    model->BSIM4paramCPU.BSIM4drainConductanceArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4drainConductanceArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4drainConductanceArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gstotsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gstotsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gstotsArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gdtotsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gdtotsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gdtotsArray, size, double, status)

    model->BSIM4paramCPU.BSIM4vfbzbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4vfbzbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4vfbzbArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gIgssArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gIgssArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gIgssArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gIgddArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gIgddArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gIgddArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gIgbsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gIgbsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gIgbsArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gIgcssArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gIgcssArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gIgcssArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gIgcdsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gIgcdsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gIgcdsArray, size, double, status)

    model->BSIM4paramCPU.BSIM4noiGd0Array = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4noiGd0Array), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4noiGd0Array, size, double, status)

    model->BSIM4paramCPU.BSIM4cqdbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cqdbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cqdbArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cqsbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cqsbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cqsbArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cqgbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cqgbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cqgbArray, size, double, status)

    model->BSIM4paramCPU.BSIM4qchqsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4qchqsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4qchqsArray, size, double, status)

    model->BSIM4paramCPU.BSIM4cqbbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4cqbbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4cqbbArray, size, double, status)

    model->BSIM4paramCPU.BSIM4taunetArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4taunetArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4taunetArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gtgArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gtgArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gtgArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gtdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gtdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gtdArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gtsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gtsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gtsArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gtbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gtbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gtbArray, size, double, status)

    model->BSIM4paramCPU.BSIM4mArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4mArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4mArray, size, double, status)

    model->BSIM4paramCPU.BSIM4grbpdArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4grbpdArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4grbpdArray, size, double, status)

    model->BSIM4paramCPU.BSIM4grbdbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4grbdbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4grbdbArray, size, double, status)

    model->BSIM4paramCPU.BSIM4grbpbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4grbpbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4grbpbArray, size, double, status)

    model->BSIM4paramCPU.BSIM4grbpsArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4grbpsArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4grbpsArray, size, double, status)

    model->BSIM4paramCPU.BSIM4grbsbArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4grbsbArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4grbsbArray, size, double, status)

    model->BSIM4paramCPU.BSIM4dNodePrimeRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4dNodePrimeRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4dNodePrimeRHSValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gNodePrimeRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gNodePrimeRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gNodePrimeRHSValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gNodeExtRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gNodeExtRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gNodeExtRHSValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4gNodeMidRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gNodeMidRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gNodeMidRHSValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4bNodePrimeRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4bNodePrimeRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4bNodePrimeRHSValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4sNodePrimeRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4sNodePrimeRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4sNodePrimeRHSValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4dbNodeRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4dbNodeRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4dbNodeRHSValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4sbNodeRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4sbNodeRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4sbNodeRHSValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4dNodeRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4dNodeRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4dNodeRHSValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4sNodeRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4sNodeRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4sNodeRHSValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4qNodeRHSValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4qNodeRHSValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4qNodeRHSValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4GEgeValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4GEgeValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4GEgeValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4GPgeValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4GPgeValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4GPgeValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4GEgpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4GEgpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4GEgpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4GPgpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4GPgpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4GPgpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4GPdpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4GPdpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4GPdpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4GPspValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4GPspValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4GPspValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4GPbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4GPbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4GPbpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4GEdpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4GEdpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4GEdpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4GEspValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4GEspValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4GEspValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4GEbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4GEbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4GEbpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4GEgmValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4GEgmValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4GEgmValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4GMgeValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4GMgeValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4GMgeValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4GMgmValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4GMgmValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4GMgmValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4GMdpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4GMdpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4GMdpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4GMgpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4GMgpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4GMgpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4GMspValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4GMspValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4GMspValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4GMbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4GMbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4GMbpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DPgmValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DPgmValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DPgmValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4GPgmValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4GPgmValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4GPgmValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SPgmValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SPgmValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SPgmValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4BPgmValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4BPgmValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4BPgmValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DgpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DgpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DgpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DspValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DspValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DspValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DbpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SdpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SdpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SdpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SgpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SgpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SgpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SbpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DPdpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DPdpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DPdpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DPdValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DPdValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DPdValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DPgpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DPgpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DPgpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DPspValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DPspValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DPspValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DPbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DPbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DPbpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DdpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DdpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DdpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DdValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DdValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DdValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SPdpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SPdpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SPdpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SPgpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SPgpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SPgpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SPspValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SPspValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SPspValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SPsValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SPsValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SPsValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SPbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SPbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SPbpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SspValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SspValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SspValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SsValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SsValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SsValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4BPdpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4BPdpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4BPdpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4BPgpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4BPgpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4BPgpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4BPspValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4BPspValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4BPspValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4BPbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4BPbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4BPbpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DPdbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DPdbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DPdbValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SPsbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SPsbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SPsbValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DBdpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DBdpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DBdpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DBdbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DBdbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DBdbValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DBbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DBbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DBbpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DBbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DBbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DBbValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4BPdbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4BPdbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4BPdbValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4BPbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4BPbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4BPbValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4BPsbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4BPsbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4BPsbValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4BPbpIFValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4BPbpIFValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4BPbpIFValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SBspValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SBspValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SBspValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SBbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SBbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SBbpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SBbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SBbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SBbValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SBsbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SBsbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SBsbValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4BdbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4BdbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4BdbValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4BbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4BbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4BbpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4BsbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4BsbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4BsbValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4BbValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4BbValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4BbValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4QqValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4QqValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4QqValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4QgpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4QgpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4QgpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4QdpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4QdpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4QdpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4QspValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4QspValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4QspValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4QbpValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4QbpValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4QbpValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4DPqValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4DPqValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4DPqValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4SPqValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4SPqValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4SPqValueArray, size, double, status)

    model->BSIM4paramCPU.BSIM4GPqValueArray = (double *) malloc (size * sizeof(double)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4GPqValueArray), size * sizeof(double)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4GPqValueArray, size, double, status)

    /* INT */
    model->BSIM4paramCPU.BSIM4offArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4offArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4offArray, size, int, status)

    model->BSIM4paramCPU.BSIM4dNodePrimeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4dNodePrimeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4dNodePrimeArray, size, int, status)

    model->BSIM4paramCPU.BSIM4sNodePrimeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4sNodePrimeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4sNodePrimeArray, size, int, status)

    model->BSIM4paramCPU.BSIM4gNodePrimeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gNodePrimeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gNodePrimeArray, size, int, status)

    model->BSIM4paramCPU.BSIM4bNodePrimeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4bNodePrimeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4bNodePrimeArray, size, int, status)

    model->BSIM4paramCPU.BSIM4gNodeExtArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gNodeExtArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gNodeExtArray, size, int, status)

    model->BSIM4paramCPU.BSIM4gNodeMidArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4gNodeMidArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4gNodeMidArray, size, int, status)

    model->BSIM4paramCPU.BSIM4dbNodeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4dbNodeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4dbNodeArray, size, int, status)

    model->BSIM4paramCPU.BSIM4sbNodeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4sbNodeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4sbNodeArray, size, int, status)

    model->BSIM4paramCPU.BSIM4sNodeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4sNodeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4sNodeArray, size, int, status)

    model->BSIM4paramCPU.BSIM4dNodeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4dNodeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4dNodeArray, size, int, status)

    model->BSIM4paramCPU.BSIM4qNodeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4qNodeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4qNodeArray, size, int, status)

    model->BSIM4paramCPU.BSIM4rbodyModArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4rbodyModArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4rbodyModArray, size, int, status)

    model->BSIM4paramCPU.BSIM4modeArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4modeArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4modeArray, size, int, status)

    model->BSIM4paramCPU.BSIM4rgateModArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4rgateModArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4rgateModArray, size, int, status)

    model->BSIM4paramCPU.BSIM4trnqsModArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4trnqsModArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4trnqsModArray, size, int, status)

    model->BSIM4paramCPU.BSIM4acnqsModArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4acnqsModArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4acnqsModArray, size, int, status)

    model->BSIM4paramCPU.BSIM4statesArray = (int *) malloc (size * sizeof(int)) ;
    status = cudaMalloc ((void **)&(model->BSIM4paramGPU.d_BSIM4statesArray), size * sizeof(int)) ;
    CUDAMALLOCCHECK (model->BSIM4paramGPU.d_BSIM4statesArray, size, int, status)

    return (OK) ;
}
