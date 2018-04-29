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
        fprintf (stderr, "cuBSIM4v7temp routine...\n") ; \
        fprintf (stderr, "Error: cudaMalloc failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

/* cudaMemcpy MACRO to check it for errors --> CUDAMEMCPYCHECK(name of pointer, dimension, type, status) */
#define CUDAMEMCPYCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuBSIM4v7temp routine...\n") ; \
        fprintf (stderr, "Error: cudaMemcpy failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

int
cuBSIM4v7temp
(
GENmodel *inModel
)
{
    int i ;
    long unsigned int size ;
    cudaError_t status ;
    BSIM4v7model *model = (BSIM4v7model *)inModel ;
    BSIM4v7instance *here ;

    size = (long unsigned int) model->gen.GENnInstances;

    /* Special case here->d_pParam */
    model->pParamHost = (struct bsim4SizeDependParam **) malloc (size * sizeof(struct bsim4SizeDependParam *)) ;
    status = cudaMalloc ((void **)&(model->d_pParam), size * sizeof(struct bsim4SizeDependParam *)) ;
    CUDAMALLOCCHECK (model->d_pParam, size, struct bsim4SizeDependParam *, status)

    i = 0 ;

    for (here = BSIM4v7instances(model); here != NULL ; here = BSIM4v7nextInstance(here))
    {
        if (here->pParam != NULL)
        {
            status = cudaMalloc ((void **)&(model->pParamHost [i]), sizeof(struct bsim4SizeDependParam)) ;
            CUDAMALLOCCHECK (model->pParamHost [i], 1, struct bsim4SizeDependParam, status)

            status = cudaMemcpy (model->pParamHost [i], here->pParam, sizeof(struct bsim4SizeDependParam), cudaMemcpyHostToDevice) ;
            CUDAMEMCPYCHECK(model->pParamHost [i], 1, struct bsim4SizeDependParam, status)
        }
        else
            model->pParamHost [i] = NULL ;

        i++ ;
    }

    /* Structure pointer vectors in GPU */
    status = cudaMemcpy (model->d_pParam, model->pParamHost, size * sizeof(struct bsim4SizeDependParam *), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->d_pParam, size, struct bsim4SizeDependParam *, status)
    /* -------------------------------- */

    /* DOUBLE */
    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gbsRWArray, model->BSIM4v7paramCPU.BSIM4v7gbsRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gbsRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cbsRWArray, model->BSIM4v7paramCPU.BSIM4v7cbsRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cbsRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gbdRWArray, model->BSIM4v7paramCPU.BSIM4v7gbdRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gbdRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cbdRWArray, model->BSIM4v7paramCPU.BSIM4v7cbdRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cbdRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7vonRWArray, model->BSIM4v7paramCPU.BSIM4v7vonRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vonRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7vdsatRWArray, model->BSIM4v7paramCPU.BSIM4v7vdsatRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vdsatRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7csubRWArray, model->BSIM4v7paramCPU.BSIM4v7csubRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7csubRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gdsRWArray, model->BSIM4v7paramCPU.BSIM4v7gdsRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gdsRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gmRWArray, model->BSIM4v7paramCPU.BSIM4v7gmRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gmRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gmbsRWArray, model->BSIM4v7paramCPU.BSIM4v7gmbsRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gmbsRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gcrgRWArray, model->BSIM4v7paramCPU.BSIM4v7gcrgRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gcrgRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7IgidlRWArray, model->BSIM4v7paramCPU.BSIM4v7IgidlRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IgidlRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7IgislRWArray, model->BSIM4v7paramCPU.BSIM4v7IgislRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IgislRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7IgcsRWArray, model->BSIM4v7paramCPU.BSIM4v7IgcsRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IgcsRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7IgcdRWArray, model->BSIM4v7paramCPU.BSIM4v7IgcdRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IgcdRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7IgsRWArray, model->BSIM4v7paramCPU.BSIM4v7IgsRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IgsRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7IgdRWArray, model->BSIM4v7paramCPU.BSIM4v7IgdRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IgdRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7IgbRWArray, model->BSIM4v7paramCPU.BSIM4v7IgbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IgbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cdRWArray, model->BSIM4v7paramCPU.BSIM4v7cdRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cdRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7qinvRWArray, model->BSIM4v7paramCPU.BSIM4v7qinvRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7qinvRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cggbRWArray, model->BSIM4v7paramCPU.BSIM4v7cggbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cggbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cgsbRWArray, model->BSIM4v7paramCPU.BSIM4v7cgsbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cgsbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cgdbRWArray, model->BSIM4v7paramCPU.BSIM4v7cgdbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cgdbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cdgbRWArray, model->BSIM4v7paramCPU.BSIM4v7cdgbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cdgbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cdsbRWArray, model->BSIM4v7paramCPU.BSIM4v7cdsbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cdsbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cddbRWArray, model->BSIM4v7paramCPU.BSIM4v7cddbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cddbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cbgbRWArray, model->BSIM4v7paramCPU.BSIM4v7cbgbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cbgbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cbsbRWArray, model->BSIM4v7paramCPU.BSIM4v7cbsbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cbsbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cbdbRWArray, model->BSIM4v7paramCPU.BSIM4v7cbdbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cbdbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7csgbRWArray, model->BSIM4v7paramCPU.BSIM4v7csgbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7csgbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cssbRWArray, model->BSIM4v7paramCPU.BSIM4v7cssbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cssbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7csdbRWArray, model->BSIM4v7paramCPU.BSIM4v7csdbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7csdbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cgbbRWArray, model->BSIM4v7paramCPU.BSIM4v7cgbbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cgbbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7csbbRWArray, model->BSIM4v7paramCPU.BSIM4v7csbbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7csbbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cdbbRWArray, model->BSIM4v7paramCPU.BSIM4v7cdbbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cdbbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cbbbRWArray, model->BSIM4v7paramCPU.BSIM4v7cbbbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cbbbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gtauRWArray, model->BSIM4v7paramCPU.BSIM4v7gtauRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gtauRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7qgateRWArray, model->BSIM4v7paramCPU.BSIM4v7qgateRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7qgateRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7qbulkRWArray, model->BSIM4v7paramCPU.BSIM4v7qbulkRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7qbulkRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7qdrnRWArray, model->BSIM4v7paramCPU.BSIM4v7qdrnRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7qdrnRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7qsrcRWArray, model->BSIM4v7paramCPU.BSIM4v7qsrcRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7qsrcRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7capbsRWArray, model->BSIM4v7paramCPU.BSIM4v7capbsRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7capbsRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7capbdRWArray, model->BSIM4v7paramCPU.BSIM4v7capbdRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7capbdRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7icVDSArray, model->BSIM4v7paramCPU.BSIM4v7icVDSArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7icVDSArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7icVGSArray, model->BSIM4v7paramCPU.BSIM4v7icVGSArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7icVGSArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7icVBSArray, model->BSIM4v7paramCPU.BSIM4v7icVBSArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7icVBSArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7vth0Array, model->BSIM4v7paramCPU.BSIM4v7vth0Array, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vth0Array, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gbbsArray, model->BSIM4v7paramCPU.BSIM4v7gbbsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gbbsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7ggidlbArray, model->BSIM4v7paramCPU.BSIM4v7ggidlbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7ggidlbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gbgsArray, model->BSIM4v7paramCPU.BSIM4v7gbgsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gbgsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7ggidlgArray, model->BSIM4v7paramCPU.BSIM4v7ggidlgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7ggidlgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gbdsArray, model->BSIM4v7paramCPU.BSIM4v7gbdsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gbdsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7ggidldArray, model->BSIM4v7paramCPU.BSIM4v7ggidldArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7ggidldArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7ggislsArray, model->BSIM4v7paramCPU.BSIM4v7ggislsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7ggislsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7ggislgArray, model->BSIM4v7paramCPU.BSIM4v7ggislgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7ggislgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7ggislbArray, model->BSIM4v7paramCPU.BSIM4v7ggislbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7ggislbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gIgsgArray, model->BSIM4v7paramCPU.BSIM4v7gIgsgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgsgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gIgcsgArray, model->BSIM4v7paramCPU.BSIM4v7gIgcsgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgcsgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gIgcsdArray, model->BSIM4v7paramCPU.BSIM4v7gIgcsdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgcsdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gIgcsbArray, model->BSIM4v7paramCPU.BSIM4v7gIgcsbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgcsbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gIgdgArray, model->BSIM4v7paramCPU.BSIM4v7gIgdgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgdgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gIgcdgArray, model->BSIM4v7paramCPU.BSIM4v7gIgcdgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgcdgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gIgcddArray, model->BSIM4v7paramCPU.BSIM4v7gIgcddArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgcddArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gIgcdbArray, model->BSIM4v7paramCPU.BSIM4v7gIgcdbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgcdbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gIgbgArray, model->BSIM4v7paramCPU.BSIM4v7gIgbgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgbgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gIgbdArray, model->BSIM4v7paramCPU.BSIM4v7gIgbdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgbdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gIgbbArray, model->BSIM4v7paramCPU.BSIM4v7gIgbbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgbbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7ggidlsArray, model->BSIM4v7paramCPU.BSIM4v7ggidlsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7ggidlsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7ggisldArray, model->BSIM4v7paramCPU.BSIM4v7ggisldArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7ggisldArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gstotArray, model->BSIM4v7paramCPU.BSIM4v7gstotArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gstotArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gstotdArray, model->BSIM4v7paramCPU.BSIM4v7gstotdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gstotdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gstotgArray, model->BSIM4v7paramCPU.BSIM4v7gstotgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gstotgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gstotbArray, model->BSIM4v7paramCPU.BSIM4v7gstotbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gstotbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gdtotArray, model->BSIM4v7paramCPU.BSIM4v7gdtotArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gdtotArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gdtotdArray, model->BSIM4v7paramCPU.BSIM4v7gdtotdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gdtotdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gdtotgArray, model->BSIM4v7paramCPU.BSIM4v7gdtotgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gdtotgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gdtotbArray, model->BSIM4v7paramCPU.BSIM4v7gdtotbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gdtotbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cgdoArray, model->BSIM4v7paramCPU.BSIM4v7cgdoArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cgdoArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7qgdoArray, model->BSIM4v7paramCPU.BSIM4v7qgdoArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7qgdoArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cgsoArray, model->BSIM4v7paramCPU.BSIM4v7cgsoArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cgsoArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7qgsoArray, model->BSIM4v7paramCPU.BSIM4v7qgsoArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7qgsoArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7AseffArray, model->BSIM4v7paramCPU.BSIM4v7AseffArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7AseffArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7PseffArray, model->BSIM4v7paramCPU.BSIM4v7PseffArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7PseffArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7nfArray, model->BSIM4v7paramCPU.BSIM4v7nfArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7nfArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7XExpBVSArray, model->BSIM4v7paramCPU.BSIM4v7XExpBVSArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7XExpBVSArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7vjsmFwdArray, model->BSIM4v7paramCPU.BSIM4v7vjsmFwdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vjsmFwdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7IVjsmFwdArray, model->BSIM4v7paramCPU.BSIM4v7IVjsmFwdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IVjsmFwdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7vjsmRevArray, model->BSIM4v7paramCPU.BSIM4v7vjsmRevArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vjsmRevArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7IVjsmRevArray, model->BSIM4v7paramCPU.BSIM4v7IVjsmRevArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IVjsmRevArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7SslpRevArray, model->BSIM4v7paramCPU.BSIM4v7SslpRevArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SslpRevArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7SslpFwdArray, model->BSIM4v7paramCPU.BSIM4v7SslpFwdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SslpFwdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7AdeffArray, model->BSIM4v7paramCPU.BSIM4v7AdeffArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7AdeffArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7PdeffArray, model->BSIM4v7paramCPU.BSIM4v7PdeffArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7PdeffArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7XExpBVDArray, model->BSIM4v7paramCPU.BSIM4v7XExpBVDArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7XExpBVDArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7vjdmFwdArray, model->BSIM4v7paramCPU.BSIM4v7vjdmFwdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vjdmFwdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7IVjdmFwdArray, model->BSIM4v7paramCPU.BSIM4v7IVjdmFwdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IVjdmFwdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7vjdmRevArray, model->BSIM4v7paramCPU.BSIM4v7vjdmRevArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vjdmRevArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7IVjdmRevArray, model->BSIM4v7paramCPU.BSIM4v7IVjdmRevArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IVjdmRevArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7DslpRevArray, model->BSIM4v7paramCPU.BSIM4v7DslpRevArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DslpRevArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7DslpFwdArray, model->BSIM4v7paramCPU.BSIM4v7DslpFwdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DslpFwdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7SjctTempRevSatCurArray, model->BSIM4v7paramCPU.BSIM4v7SjctTempRevSatCurArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SjctTempRevSatCurArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7SswTempRevSatCurArray, model->BSIM4v7paramCPU.BSIM4v7SswTempRevSatCurArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SswTempRevSatCurArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7SswgTempRevSatCurArray, model->BSIM4v7paramCPU.BSIM4v7SswgTempRevSatCurArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7SswgTempRevSatCurArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7DjctTempRevSatCurArray, model->BSIM4v7paramCPU.BSIM4v7DjctTempRevSatCurArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DjctTempRevSatCurArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7DswTempRevSatCurArray, model->BSIM4v7paramCPU.BSIM4v7DswTempRevSatCurArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DswTempRevSatCurArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7DswgTempRevSatCurArray, model->BSIM4v7paramCPU.BSIM4v7DswgTempRevSatCurArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7DswgTempRevSatCurArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7vbscArray, model->BSIM4v7paramCPU.BSIM4v7vbscArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vbscArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7thetavthArray, model->BSIM4v7paramCPU.BSIM4v7thetavthArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7thetavthArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7eta0Array, model->BSIM4v7paramCPU.BSIM4v7eta0Array, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7eta0Array, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7k2oxArray, model->BSIM4v7paramCPU.BSIM4v7k2oxArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7k2oxArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7nstarArray, model->BSIM4v7paramCPU.BSIM4v7nstarArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7nstarArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7vfbArray, model->BSIM4v7paramCPU.BSIM4v7vfbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vfbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7vgs_effArray, model->BSIM4v7paramCPU.BSIM4v7vgs_effArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vgs_effArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7vgd_effArray, model->BSIM4v7paramCPU.BSIM4v7vgd_effArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vgd_effArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7dvgs_eff_dvgArray, model->BSIM4v7paramCPU.BSIM4v7dvgs_eff_dvgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7dvgs_eff_dvgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7dvgd_eff_dvgArray, model->BSIM4v7paramCPU.BSIM4v7dvgd_eff_dvgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7dvgd_eff_dvgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7VgsteffArray, model->BSIM4v7paramCPU.BSIM4v7VgsteffArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7VgsteffArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7grdswArray, model->BSIM4v7paramCPU.BSIM4v7grdswArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7grdswArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7AbulkArray, model->BSIM4v7paramCPU.BSIM4v7AbulkArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7AbulkArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7vtfbphi1Array, model->BSIM4v7paramCPU.BSIM4v7vtfbphi1Array, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vtfbphi1Array, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7ueffArray, model->BSIM4v7paramCPU.BSIM4v7ueffArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7ueffArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7u0tempArray, model->BSIM4v7paramCPU.BSIM4v7u0tempArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7u0tempArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7vsattempArray, model->BSIM4v7paramCPU.BSIM4v7vsattempArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vsattempArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7EsatLArray, model->BSIM4v7paramCPU.BSIM4v7EsatLArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7EsatLArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7VdseffArray, model->BSIM4v7paramCPU.BSIM4v7VdseffArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7VdseffArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7vtfbphi2Array, model->BSIM4v7paramCPU.BSIM4v7vtfbphi2Array, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vtfbphi2Array, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7CoxeffArray, model->BSIM4v7paramCPU.BSIM4v7CoxeffArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7CoxeffArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7AbovVgst2VtmArray, model->BSIM4v7paramCPU.BSIM4v7AbovVgst2VtmArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7AbovVgst2VtmArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7IdovVdsArray, model->BSIM4v7paramCPU.BSIM4v7IdovVdsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7IdovVdsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gcrgdArray, model->BSIM4v7paramCPU.BSIM4v7gcrgdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gcrgdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gcrgbArray, model->BSIM4v7paramCPU.BSIM4v7gcrgbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gcrgbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gcrggArray, model->BSIM4v7paramCPU.BSIM4v7gcrggArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gcrggArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7grgeltdArray, model->BSIM4v7paramCPU.BSIM4v7grgeltdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7grgeltdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gcrgsArray, model->BSIM4v7paramCPU.BSIM4v7gcrgsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gcrgsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7sourceConductanceArray, model->BSIM4v7paramCPU.BSIM4v7sourceConductanceArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7sourceConductanceArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7drainConductanceArray, model->BSIM4v7paramCPU.BSIM4v7drainConductanceArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7drainConductanceArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gstotsArray, model->BSIM4v7paramCPU.BSIM4v7gstotsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gstotsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gdtotsArray, model->BSIM4v7paramCPU.BSIM4v7gdtotsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gdtotsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7vfbzbArray, model->BSIM4v7paramCPU.BSIM4v7vfbzbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7vfbzbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gIgssArray, model->BSIM4v7paramCPU.BSIM4v7gIgssArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgssArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gIgddArray, model->BSIM4v7paramCPU.BSIM4v7gIgddArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgddArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gIgbsArray, model->BSIM4v7paramCPU.BSIM4v7gIgbsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgbsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gIgcssArray, model->BSIM4v7paramCPU.BSIM4v7gIgcssArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgcssArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gIgcdsArray, model->BSIM4v7paramCPU.BSIM4v7gIgcdsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gIgcdsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7noiGd0Array, model->BSIM4v7paramCPU.BSIM4v7noiGd0Array, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7noiGd0Array, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cqdbArray, model->BSIM4v7paramCPU.BSIM4v7cqdbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cqdbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cqsbArray, model->BSIM4v7paramCPU.BSIM4v7cqsbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cqsbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cqgbArray, model->BSIM4v7paramCPU.BSIM4v7cqgbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cqgbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7qchqsArray, model->BSIM4v7paramCPU.BSIM4v7qchqsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7qchqsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7cqbbArray, model->BSIM4v7paramCPU.BSIM4v7cqbbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7cqbbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7taunetArray, model->BSIM4v7paramCPU.BSIM4v7taunetArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7taunetArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gtgArray, model->BSIM4v7paramCPU.BSIM4v7gtgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gtgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gtdArray, model->BSIM4v7paramCPU.BSIM4v7gtdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gtdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gtsArray, model->BSIM4v7paramCPU.BSIM4v7gtsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gtsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gtbArray, model->BSIM4v7paramCPU.BSIM4v7gtbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gtbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7mArray, model->BSIM4v7paramCPU.BSIM4v7mArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7mArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7grbpdArray, model->BSIM4v7paramCPU.BSIM4v7grbpdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7grbpdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7grbdbArray, model->BSIM4v7paramCPU.BSIM4v7grbdbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7grbdbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7grbpbArray, model->BSIM4v7paramCPU.BSIM4v7grbpbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7grbpbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7grbpsArray, model->BSIM4v7paramCPU.BSIM4v7grbpsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7grbpsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7grbsbArray, model->BSIM4v7paramCPU.BSIM4v7grbsbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7grbsbArray, size, double, status)

    /* INT */
    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7offArray, model->BSIM4v7paramCPU.BSIM4v7offArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7offArray, size, int, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7dNodePrimeArray, model->BSIM4v7paramCPU.BSIM4v7dNodePrimeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7dNodePrimeArray, size, int, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7sNodePrimeArray, model->BSIM4v7paramCPU.BSIM4v7sNodePrimeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7sNodePrimeArray, size, int, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gNodePrimeArray, model->BSIM4v7paramCPU.BSIM4v7gNodePrimeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gNodePrimeArray, size, int, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7bNodePrimeArray, model->BSIM4v7paramCPU.BSIM4v7bNodePrimeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7bNodePrimeArray, size, int, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gNodeExtArray, model->BSIM4v7paramCPU.BSIM4v7gNodeExtArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gNodeExtArray, size, int, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7gNodeMidArray, model->BSIM4v7paramCPU.BSIM4v7gNodeMidArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7gNodeMidArray, size, int, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7dbNodeArray, model->BSIM4v7paramCPU.BSIM4v7dbNodeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7dbNodeArray, size, int, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7sbNodeArray, model->BSIM4v7paramCPU.BSIM4v7sbNodeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7sbNodeArray, size, int, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7sNodeArray, model->BSIM4v7paramCPU.BSIM4v7sNodeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7sNodeArray, size, int, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7dNodeArray, model->BSIM4v7paramCPU.BSIM4v7dNodeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7dNodeArray, size, int, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7qNodeArray, model->BSIM4v7paramCPU.BSIM4v7qNodeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7qNodeArray, size, int, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7rbodyModArray, model->BSIM4v7paramCPU.BSIM4v7rbodyModArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7rbodyModArray, size, int, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7modeArray, model->BSIM4v7paramCPU.BSIM4v7modeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7modeArray, size, int, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7rgateModArray, model->BSIM4v7paramCPU.BSIM4v7rgateModArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7rgateModArray, size, int, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7trnqsModArray, model->BSIM4v7paramCPU.BSIM4v7trnqsModArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7trnqsModArray, size, int, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7acnqsModArray, model->BSIM4v7paramCPU.BSIM4v7acnqsModArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.d_BSIM4v7acnqsModArray, size, int, status)

    status = cudaMemcpy (model->BSIM4v7paramGPU.d_BSIM4v7statesArray, model->BSIM4v7paramCPU.BSIM4v7statesArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4v7paramGPU.BSIM4v7statesArray, size, int, status)

    return (OK) ;
}
