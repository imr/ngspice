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

/* cudaMemcpy MACRO to check it for errors --> CUDAMEMCPYCHECK(name of pointer, dimension, type, status) */
#define CUDAMEMCPYCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuMUTtemp routine...\n") ; \
        fprintf (stderr, "Error: cudaMemcpy failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

int
cuMUTtemp
(
GENmodel *inModel
)
{
    long unsigned int size ;
    cudaError_t status ;
    MUTmodel *model = (MUTmodel *)inModel ;

    size = (long unsigned int) model->gen.GENnInstances;

    /* DOUBLE */
    status = cudaMemcpy (model->MUTparamGPU.d_MUTfactorArray, model->MUTparamCPU.MUTfactorArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK(model->MUTparamGPU.d_MUTfactorArray, size, double, status)

    /* INT */
    status = cudaMemcpy (model->MUTparamGPU.d_MUTflux1Array, model->MUTparamCPU.MUTflux1Array, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK(model->MUTparamGPU.d_MUTflux1Array, size, int, status)

    status = cudaMemcpy (model->MUTparamGPU.d_MUTflux2Array, model->MUTparamCPU.MUTflux2Array, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK(model->MUTparamGPU.d_MUTflux2Array, size, int, status)

    status = cudaMemcpy (model->MUTparamGPU.d_MUTbrEq1Array, model->MUTparamCPU.MUTbrEq1Array, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK(model->MUTparamGPU.d_MUTbrEq1Array, size, int, status)

    status = cudaMemcpy (model->MUTparamGPU.d_MUTbrEq2Array, model->MUTparamCPU.MUTbrEq2Array, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK(model->MUTparamGPU.MUTbrEq2Array, size, int, status)

    return (OK) ;
}
