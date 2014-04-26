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
#include "isrcdefs.h"
#include "ngspice/CUSPICE/CUSPICE.h"

/* cudaMemcpy MACRO to check it for errors --> CUDAMEMCPYCHECK(name of pointer, dimension, type, status) */
#define CUDAMEMCPYCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuISRCtemp routine...\n") ; \
        fprintf (stderr, "Error: cudaMemcpy failed on %s size1 of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

int
cuISRCtemp
(
GENmodel *inModel
)
{
    int i ;
    long unsigned int size1, size2 ;
    cudaError_t status ;
    ISRCmodel *model = (ISRCmodel *)inModel ;
    ISRCinstance *here ;

    size1 = (long unsigned int) model->n_instances;

    i = 0 ;

    for (here = ISRCinstances(model); here != NULL ; here = ISRCnextInstance(here))
    {
        size2 = (long unsigned int)here->n_coeffs ;
        status = cudaMemcpy (model->ISRCparamCPU.ISRCcoeffsArray [i], model->ISRCparamCPU.ISRCcoeffsArrayHost [i], size2 * sizeof(double), cudaMemcpyHostToDevice) ;
        CUDAMEMCPYCHECK (model->ISRCparamCPU.ISRCcoeffsArray [i], size2, double, status)

        i++ ;
    }

    /* DOUBLE */
    status = cudaMemcpy (model->ISRCparamGPU.d_ISRCdcvalueArray, model->ISRCparamCPU.ISRCdcvalueArray, size1 * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK(model->ISRCparamGPU.d_ISRCdcvalueArray, size1, double, status)

    /* INT */
    status = cudaMemcpy (model->ISRCparamGPU.d_ISRCdcGivenArray, model->ISRCparamCPU.ISRCdcGivenArray, size1 * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK(model->ISRCparamGPU.d_ISRCdcGivenArray, size1, int, status)

    status = cudaMemcpy (model->ISRCparamGPU.d_ISRCfunctionTypeArray, model->ISRCparamCPU.ISRCfunctionTypeArray, size1 * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK(model->ISRCparamGPU.d_ISRCfunctionTypeArray, size1, int, status)

    status = cudaMemcpy (model->ISRCparamGPU.d_ISRCfunctionOrderArray, model->ISRCparamCPU.ISRCfunctionOrderArray, size1 * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK(model->ISRCparamGPU.ISRCfunctionOrderArray, size1, int, status)

    return (OK) ;
}
