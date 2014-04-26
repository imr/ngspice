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
        fprintf (stderr, "cuINDtemp routine...\n") ; \
        fprintf (stderr, "Error: cudaMemcpy failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

int
cuINDtemp
(
GENmodel *inModel
)
{
    long unsigned int size ;
    cudaError_t status ;
    INDmodel *model = (INDmodel *)inModel ;

    size = (long unsigned int) model->n_instances;

    /* DOUBLE */
    status = cudaMemcpy (model->INDparamGPU.d_INDinitCondArray, model->INDparamCPU.INDinitCondArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK(model->INDparamGPU.d_INDinitCondArray, size, double, status)

    status = cudaMemcpy (model->INDparamGPU.d_INDinductArray, model->INDparamCPU.INDinductArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK(model->INDparamGPU.d_INDinductArray, size, double, status)

    /* INT */
    status = cudaMemcpy (model->INDparamGPU.d_INDbrEqArray, model->INDparamCPU.INDbrEqArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK(model->INDparamGPU.d_INDbrEqArray, size, int, status)

    status = cudaMemcpy (model->INDparamGPU.d_INDstateArray, model->INDparamCPU.INDstateArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK(model->INDparamGPU.INDstateArray, size, int, status)

    return (OK) ;
}
