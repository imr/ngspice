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
        fprintf (stderr, "cuBSIM4temp routine...\n") ; \
        fprintf (stderr, "Error: cudaMalloc failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

/* cudaMemcpy MACRO to check it for errors --> CUDAMEMCPYCHECK(name of pointer, dimension, type, status) */
#define CUDAMEMCPYCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuBSIM4temp routine...\n") ; \
        fprintf (stderr, "Error: cudaMemcpy failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

int
cuBSIM4temp
(
GENmodel *inModel
)
{
    int i ;
    long unsigned int size ;
    cudaError_t status ;
    BSIM4model *model = (BSIM4model *)inModel ;
    BSIM4instance *here ;

    size = (long unsigned int)model->n_instances ;

    /* Special case here->d_pParam */
    model->pParamHost = (struct bsim4SizeDependParam **) malloc (size * sizeof(struct bsim4SizeDependParam *)) ;
    status = cudaMalloc ((void **)&(model->d_pParam), size * sizeof(struct bsim4SizeDependParam *)) ;
    CUDAMALLOCCHECK (model->d_pParam, size, struct bsim4SizeDependParam *, status)

    i = 0 ;

    for (here = model->BSIM4instances ; here != NULL ; here = here->BSIM4nextInstance)
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
    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gbsRWArray, model->BSIM4paramCPU.BSIM4gbsRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gbsRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cbsRWArray, model->BSIM4paramCPU.BSIM4cbsRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cbsRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gbdRWArray, model->BSIM4paramCPU.BSIM4gbdRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gbdRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cbdRWArray, model->BSIM4paramCPU.BSIM4cbdRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cbdRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4vonRWArray, model->BSIM4paramCPU.BSIM4vonRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4vonRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4vdsatRWArray, model->BSIM4paramCPU.BSIM4vdsatRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4vdsatRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4csubRWArray, model->BSIM4paramCPU.BSIM4csubRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4csubRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gdsRWArray, model->BSIM4paramCPU.BSIM4gdsRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gdsRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gmRWArray, model->BSIM4paramCPU.BSIM4gmRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gmRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gmbsRWArray, model->BSIM4paramCPU.BSIM4gmbsRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gmbsRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gcrgRWArray, model->BSIM4paramCPU.BSIM4gcrgRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gcrgRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4IgidlRWArray, model->BSIM4paramCPU.BSIM4IgidlRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4IgidlRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4IgislRWArray, model->BSIM4paramCPU.BSIM4IgislRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4IgislRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4IgcsRWArray, model->BSIM4paramCPU.BSIM4IgcsRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4IgcsRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4IgcdRWArray, model->BSIM4paramCPU.BSIM4IgcdRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4IgcdRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4IgsRWArray, model->BSIM4paramCPU.BSIM4IgsRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4IgsRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4IgdRWArray, model->BSIM4paramCPU.BSIM4IgdRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4IgdRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4IgbRWArray, model->BSIM4paramCPU.BSIM4IgbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4IgbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cdRWArray, model->BSIM4paramCPU.BSIM4cdRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cdRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4qinvRWArray, model->BSIM4paramCPU.BSIM4qinvRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4qinvRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cggbRWArray, model->BSIM4paramCPU.BSIM4cggbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cggbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cgsbRWArray, model->BSIM4paramCPU.BSIM4cgsbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cgsbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cgdbRWArray, model->BSIM4paramCPU.BSIM4cgdbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cgdbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cdgbRWArray, model->BSIM4paramCPU.BSIM4cdgbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cdgbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cdsbRWArray, model->BSIM4paramCPU.BSIM4cdsbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cdsbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cddbRWArray, model->BSIM4paramCPU.BSIM4cddbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cddbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cbgbRWArray, model->BSIM4paramCPU.BSIM4cbgbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cbgbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cbsbRWArray, model->BSIM4paramCPU.BSIM4cbsbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cbsbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cbdbRWArray, model->BSIM4paramCPU.BSIM4cbdbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cbdbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4csgbRWArray, model->BSIM4paramCPU.BSIM4csgbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4csgbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cssbRWArray, model->BSIM4paramCPU.BSIM4cssbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cssbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4csdbRWArray, model->BSIM4paramCPU.BSIM4csdbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4csdbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cgbbRWArray, model->BSIM4paramCPU.BSIM4cgbbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cgbbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4csbbRWArray, model->BSIM4paramCPU.BSIM4csbbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4csbbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cdbbRWArray, model->BSIM4paramCPU.BSIM4cdbbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cdbbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cbbbRWArray, model->BSIM4paramCPU.BSIM4cbbbRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cbbbRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gtauRWArray, model->BSIM4paramCPU.BSIM4gtauRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gtauRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4qgateRWArray, model->BSIM4paramCPU.BSIM4qgateRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4qgateRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4qbulkRWArray, model->BSIM4paramCPU.BSIM4qbulkRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4qbulkRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4qdrnRWArray, model->BSIM4paramCPU.BSIM4qdrnRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4qdrnRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4qsrcRWArray, model->BSIM4paramCPU.BSIM4qsrcRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4qsrcRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4capbsRWArray, model->BSIM4paramCPU.BSIM4capbsRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4capbsRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4capbdRWArray, model->BSIM4paramCPU.BSIM4capbdRWArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4capbdRWArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4icVDSArray, model->BSIM4paramCPU.BSIM4icVDSArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4icVDSArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4icVGSArray, model->BSIM4paramCPU.BSIM4icVGSArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4icVGSArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4icVBSArray, model->BSIM4paramCPU.BSIM4icVBSArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4icVBSArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4vth0Array, model->BSIM4paramCPU.BSIM4vth0Array, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4vth0Array, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gbbsArray, model->BSIM4paramCPU.BSIM4gbbsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gbbsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4ggidlbArray, model->BSIM4paramCPU.BSIM4ggidlbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4ggidlbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gbgsArray, model->BSIM4paramCPU.BSIM4gbgsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gbgsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4ggidlgArray, model->BSIM4paramCPU.BSIM4ggidlgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4ggidlgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gbdsArray, model->BSIM4paramCPU.BSIM4gbdsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gbdsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4ggidldArray, model->BSIM4paramCPU.BSIM4ggidldArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4ggidldArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4ggislsArray, model->BSIM4paramCPU.BSIM4ggislsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4ggislsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4ggislgArray, model->BSIM4paramCPU.BSIM4ggislgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4ggislgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4ggislbArray, model->BSIM4paramCPU.BSIM4ggislbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4ggislbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gIgsgArray, model->BSIM4paramCPU.BSIM4gIgsgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gIgsgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gIgcsgArray, model->BSIM4paramCPU.BSIM4gIgcsgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gIgcsgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gIgcsdArray, model->BSIM4paramCPU.BSIM4gIgcsdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gIgcsdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gIgcsbArray, model->BSIM4paramCPU.BSIM4gIgcsbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gIgcsbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gIgdgArray, model->BSIM4paramCPU.BSIM4gIgdgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gIgdgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gIgcdgArray, model->BSIM4paramCPU.BSIM4gIgcdgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gIgcdgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gIgcddArray, model->BSIM4paramCPU.BSIM4gIgcddArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gIgcddArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gIgcdbArray, model->BSIM4paramCPU.BSIM4gIgcdbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gIgcdbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gIgbgArray, model->BSIM4paramCPU.BSIM4gIgbgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gIgbgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gIgbdArray, model->BSIM4paramCPU.BSIM4gIgbdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gIgbdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gIgbbArray, model->BSIM4paramCPU.BSIM4gIgbbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gIgbbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4ggidlsArray, model->BSIM4paramCPU.BSIM4ggidlsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4ggidlsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4ggisldArray, model->BSIM4paramCPU.BSIM4ggisldArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4ggisldArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gstotArray, model->BSIM4paramCPU.BSIM4gstotArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gstotArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gstotdArray, model->BSIM4paramCPU.BSIM4gstotdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gstotdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gstotgArray, model->BSIM4paramCPU.BSIM4gstotgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gstotgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gstotbArray, model->BSIM4paramCPU.BSIM4gstotbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gstotbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gdtotArray, model->BSIM4paramCPU.BSIM4gdtotArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gdtotArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gdtotdArray, model->BSIM4paramCPU.BSIM4gdtotdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gdtotdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gdtotgArray, model->BSIM4paramCPU.BSIM4gdtotgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gdtotgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gdtotbArray, model->BSIM4paramCPU.BSIM4gdtotbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gdtotbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cgdoArray, model->BSIM4paramCPU.BSIM4cgdoArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cgdoArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4qgdoArray, model->BSIM4paramCPU.BSIM4qgdoArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4qgdoArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cgsoArray, model->BSIM4paramCPU.BSIM4cgsoArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cgsoArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4qgsoArray, model->BSIM4paramCPU.BSIM4qgsoArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4qgsoArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4AseffArray, model->BSIM4paramCPU.BSIM4AseffArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4AseffArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4PseffArray, model->BSIM4paramCPU.BSIM4PseffArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4PseffArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4nfArray, model->BSIM4paramCPU.BSIM4nfArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4nfArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4XExpBVSArray, model->BSIM4paramCPU.BSIM4XExpBVSArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4XExpBVSArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4vjsmFwdArray, model->BSIM4paramCPU.BSIM4vjsmFwdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4vjsmFwdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4IVjsmFwdArray, model->BSIM4paramCPU.BSIM4IVjsmFwdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4IVjsmFwdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4vjsmRevArray, model->BSIM4paramCPU.BSIM4vjsmRevArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4vjsmRevArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4IVjsmRevArray, model->BSIM4paramCPU.BSIM4IVjsmRevArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4IVjsmRevArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4SslpRevArray, model->BSIM4paramCPU.BSIM4SslpRevArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4SslpRevArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4SslpFwdArray, model->BSIM4paramCPU.BSIM4SslpFwdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4SslpFwdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4AdeffArray, model->BSIM4paramCPU.BSIM4AdeffArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4AdeffArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4PdeffArray, model->BSIM4paramCPU.BSIM4PdeffArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4PdeffArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4XExpBVDArray, model->BSIM4paramCPU.BSIM4XExpBVDArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4XExpBVDArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4vjdmFwdArray, model->BSIM4paramCPU.BSIM4vjdmFwdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4vjdmFwdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4IVjdmFwdArray, model->BSIM4paramCPU.BSIM4IVjdmFwdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4IVjdmFwdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4vjdmRevArray, model->BSIM4paramCPU.BSIM4vjdmRevArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4vjdmRevArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4IVjdmRevArray, model->BSIM4paramCPU.BSIM4IVjdmRevArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4IVjdmRevArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4DslpRevArray, model->BSIM4paramCPU.BSIM4DslpRevArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4DslpRevArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4DslpFwdArray, model->BSIM4paramCPU.BSIM4DslpFwdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4DslpFwdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4SjctTempRevSatCurArray, model->BSIM4paramCPU.BSIM4SjctTempRevSatCurArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4SjctTempRevSatCurArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4SswTempRevSatCurArray, model->BSIM4paramCPU.BSIM4SswTempRevSatCurArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4SswTempRevSatCurArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4SswgTempRevSatCurArray, model->BSIM4paramCPU.BSIM4SswgTempRevSatCurArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4SswgTempRevSatCurArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4DjctTempRevSatCurArray, model->BSIM4paramCPU.BSIM4DjctTempRevSatCurArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4DjctTempRevSatCurArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4DswTempRevSatCurArray, model->BSIM4paramCPU.BSIM4DswTempRevSatCurArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4DswTempRevSatCurArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4DswgTempRevSatCurArray, model->BSIM4paramCPU.BSIM4DswgTempRevSatCurArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4DswgTempRevSatCurArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4vbscArray, model->BSIM4paramCPU.BSIM4vbscArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4vbscArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4thetavthArray, model->BSIM4paramCPU.BSIM4thetavthArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4thetavthArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4eta0Array, model->BSIM4paramCPU.BSIM4eta0Array, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4eta0Array, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4k2oxArray, model->BSIM4paramCPU.BSIM4k2oxArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4k2oxArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4nstarArray, model->BSIM4paramCPU.BSIM4nstarArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4nstarArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4vfbArray, model->BSIM4paramCPU.BSIM4vfbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4vfbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4vgs_effArray, model->BSIM4paramCPU.BSIM4vgs_effArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4vgs_effArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4vgd_effArray, model->BSIM4paramCPU.BSIM4vgd_effArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4vgd_effArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4dvgs_eff_dvgArray, model->BSIM4paramCPU.BSIM4dvgs_eff_dvgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4dvgs_eff_dvgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4dvgd_eff_dvgArray, model->BSIM4paramCPU.BSIM4dvgd_eff_dvgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4dvgd_eff_dvgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4VgsteffArray, model->BSIM4paramCPU.BSIM4VgsteffArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4VgsteffArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4grdswArray, model->BSIM4paramCPU.BSIM4grdswArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4grdswArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4AbulkArray, model->BSIM4paramCPU.BSIM4AbulkArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4AbulkArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4vtfbphi1Array, model->BSIM4paramCPU.BSIM4vtfbphi1Array, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4vtfbphi1Array, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4ueffArray, model->BSIM4paramCPU.BSIM4ueffArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4ueffArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4u0tempArray, model->BSIM4paramCPU.BSIM4u0tempArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4u0tempArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4vsattempArray, model->BSIM4paramCPU.BSIM4vsattempArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4vsattempArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4EsatLArray, model->BSIM4paramCPU.BSIM4EsatLArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4EsatLArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4VdseffArray, model->BSIM4paramCPU.BSIM4VdseffArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4VdseffArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4vtfbphi2Array, model->BSIM4paramCPU.BSIM4vtfbphi2Array, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4vtfbphi2Array, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4CoxeffArray, model->BSIM4paramCPU.BSIM4CoxeffArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4CoxeffArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4AbovVgst2VtmArray, model->BSIM4paramCPU.BSIM4AbovVgst2VtmArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4AbovVgst2VtmArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4IdovVdsArray, model->BSIM4paramCPU.BSIM4IdovVdsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4IdovVdsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gcrgdArray, model->BSIM4paramCPU.BSIM4gcrgdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gcrgdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gcrgbArray, model->BSIM4paramCPU.BSIM4gcrgbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gcrgbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gcrggArray, model->BSIM4paramCPU.BSIM4gcrggArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gcrggArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4grgeltdArray, model->BSIM4paramCPU.BSIM4grgeltdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4grgeltdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gcrgsArray, model->BSIM4paramCPU.BSIM4gcrgsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gcrgsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4sourceConductanceArray, model->BSIM4paramCPU.BSIM4sourceConductanceArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4sourceConductanceArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4drainConductanceArray, model->BSIM4paramCPU.BSIM4drainConductanceArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4drainConductanceArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gstotsArray, model->BSIM4paramCPU.BSIM4gstotsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gstotsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gdtotsArray, model->BSIM4paramCPU.BSIM4gdtotsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gdtotsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4vfbzbArray, model->BSIM4paramCPU.BSIM4vfbzbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4vfbzbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gIgssArray, model->BSIM4paramCPU.BSIM4gIgssArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gIgssArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gIgddArray, model->BSIM4paramCPU.BSIM4gIgddArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gIgddArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gIgbsArray, model->BSIM4paramCPU.BSIM4gIgbsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gIgbsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gIgcssArray, model->BSIM4paramCPU.BSIM4gIgcssArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gIgcssArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gIgcdsArray, model->BSIM4paramCPU.BSIM4gIgcdsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gIgcdsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4noiGd0Array, model->BSIM4paramCPU.BSIM4noiGd0Array, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4noiGd0Array, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cqdbArray, model->BSIM4paramCPU.BSIM4cqdbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cqdbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cqsbArray, model->BSIM4paramCPU.BSIM4cqsbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cqsbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cqgbArray, model->BSIM4paramCPU.BSIM4cqgbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cqgbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4qchqsArray, model->BSIM4paramCPU.BSIM4qchqsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4qchqsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4cqbbArray, model->BSIM4paramCPU.BSIM4cqbbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4cqbbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4taunetArray, model->BSIM4paramCPU.BSIM4taunetArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4taunetArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gtgArray, model->BSIM4paramCPU.BSIM4gtgArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gtgArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gtdArray, model->BSIM4paramCPU.BSIM4gtdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gtdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gtsArray, model->BSIM4paramCPU.BSIM4gtsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gtsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gtbArray, model->BSIM4paramCPU.BSIM4gtbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gtbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4mArray, model->BSIM4paramCPU.BSIM4mArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4mArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4grbpdArray, model->BSIM4paramCPU.BSIM4grbpdArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4grbpdArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4grbdbArray, model->BSIM4paramCPU.BSIM4grbdbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4grbdbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4grbpbArray, model->BSIM4paramCPU.BSIM4grbpbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4grbpbArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4grbpsArray, model->BSIM4paramCPU.BSIM4grbpsArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4grbpsArray, size, double, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4grbsbArray, model->BSIM4paramCPU.BSIM4grbsbArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4grbsbArray, size, double, status)

    /* INT */
    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4offArray, model->BSIM4paramCPU.BSIM4offArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4offArray, size, int, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4dNodePrimeArray, model->BSIM4paramCPU.BSIM4dNodePrimeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4dNodePrimeArray, size, int, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4sNodePrimeArray, model->BSIM4paramCPU.BSIM4sNodePrimeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4sNodePrimeArray, size, int, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gNodePrimeArray, model->BSIM4paramCPU.BSIM4gNodePrimeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gNodePrimeArray, size, int, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4bNodePrimeArray, model->BSIM4paramCPU.BSIM4bNodePrimeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4bNodePrimeArray, size, int, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gNodeExtArray, model->BSIM4paramCPU.BSIM4gNodeExtArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gNodeExtArray, size, int, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4gNodeMidArray, model->BSIM4paramCPU.BSIM4gNodeMidArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4gNodeMidArray, size, int, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4dbNodeArray, model->BSIM4paramCPU.BSIM4dbNodeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4dbNodeArray, size, int, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4sbNodeArray, model->BSIM4paramCPU.BSIM4sbNodeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4sbNodeArray, size, int, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4sNodeArray, model->BSIM4paramCPU.BSIM4sNodeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4sNodeArray, size, int, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4dNodeArray, model->BSIM4paramCPU.BSIM4dNodeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4dNodeArray, size, int, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4qNodeArray, model->BSIM4paramCPU.BSIM4qNodeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4qNodeArray, size, int, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4rbodyModArray, model->BSIM4paramCPU.BSIM4rbodyModArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4rbodyModArray, size, int, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4modeArray, model->BSIM4paramCPU.BSIM4modeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4modeArray, size, int, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4rgateModArray, model->BSIM4paramCPU.BSIM4rgateModArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4rgateModArray, size, int, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4trnqsModArray, model->BSIM4paramCPU.BSIM4trnqsModArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4trnqsModArray, size, int, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4acnqsModArray, model->BSIM4paramCPU.BSIM4acnqsModArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.d_BSIM4acnqsModArray, size, int, status)

    status = cudaMemcpy (model->BSIM4paramGPU.d_BSIM4statesArray, model->BSIM4paramCPU.BSIM4statesArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK (model->BSIM4paramGPU.BSIM4statesArray, size, int, status)

    return (OK) ;
}
