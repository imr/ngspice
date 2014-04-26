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

/* cudaMemcpy MACRO to check it for errors --> CUDAMEMCPYCHECK(name of pointer, dimension, type, status) */
#define CUDAMEMCPYCHECK(a, b, c, d) \
    if (d != cudaSuccess) \
    { \
        fprintf (stderr, "cuCAPtemp routine...\n") ; \
        fprintf (stderr, "Error: cudaMemcpy failed on %s size of %d bytes\n", #a, (int)(b * sizeof(c))) ; \
        fprintf (stderr, "Error: %s = %d, %s\n", #d, d, cudaGetErrorString (d)) ; \
        return (E_NOMEM) ; \
    }

int
cuCAPtemp
(
GENmodel *inModel
)
{
    long unsigned int size ;
    cudaError_t status ;
    CAPmodel *model = (CAPmodel *)inModel ;

    size = (long unsigned int) model->n_instances;

    /* DOUBLE */
    status = cudaMemcpy (model->CAPparamGPU.d_CAPcapacArray, model->CAPparamCPU.CAPcapacArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK(model->CAPparamGPU.d_CAPcapacArray, size, double, status)

    status = cudaMemcpy (model->CAPparamGPU.d_CAPmArray, model->CAPparamCPU.CAPmArray, size * sizeof(double), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK(model->CAPparamGPU.d_CAPmArray, size, double, status)

    /* INT */
    status = cudaMemcpy (model->CAPparamGPU.d_CAPposNodeArray, model->CAPparamCPU.CAPposNodeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK(model->CAPparamGPU.d_CAPposNodeArray, size, int, status)

    status = cudaMemcpy (model->CAPparamGPU.d_CAPnegNodeArray, model->CAPparamCPU.CAPnegNodeArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK(model->CAPparamGPU.d_CAPnegNodeArray, size, int, status)

    status = cudaMemcpy (model->CAPparamGPU.d_CAPstateArray, model->CAPparamCPU.CAPstateArray, size * sizeof(int), cudaMemcpyHostToDevice) ;
    CUDAMEMCPYCHECK(model->CAPparamGPU.CAPstateArray, size, int, status)

    return (OK) ;
}
