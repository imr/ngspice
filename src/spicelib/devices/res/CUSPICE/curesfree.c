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
#include "resdefs.h"
#include "ngspice/CUSPICE/CUSPICE.h"

int
cuRESdestroy
(
GENmodel *inModel
)
{
    RESmodel *model = (RESmodel *)inModel ;

    for ( ; model != NULL ; model = RESnextModel(model))
    {
        /* DOUBLE */
        free (model->RESparamCPU.REStc1Array) ;
        cudaFree (model->RESparamGPU.d_REStc1Array) ;

        free (model->RESparamCPU.REStc2Array) ;
        cudaFree (model->RESparamGPU.d_REStc2Array) ;

        free (model->RESparamCPU.RESmArray) ;
        cudaFree (model->RESparamGPU.d_RESmArray) ;

        free (model->RESparamCPU.RESconductArray) ;
        cudaFree (model->RESparamGPU.d_RESconductArray) ;

        free (model->RESparamCPU.REStempArray) ;
        cudaFree (model->RESparamGPU.d_REStempArray) ;

        free (model->RESparamCPU.RESdtempArray) ;
        cudaFree (model->RESparamGPU.d_RESdtempArray) ;

        free (model->RESparamCPU.REScurrentArray) ;
        cudaFree (model->RESparamGPU.d_REScurrentArray) ;

        free (model->RESparamCPU.RESgValueArray) ;
        cudaFree (model->RESparamGPU.d_RESgValueArray) ;

        /* INT */
        free (model->RESparamCPU.REStc1GivenArray) ;
        cudaFree (model->RESparamGPU.d_REStc1GivenArray) ;

        free (model->RESparamCPU.REStc2GivenArray) ;
        cudaFree (model->RESparamGPU.d_REStc2GivenArray) ;

        free (model->RESparamCPU.RESmGivenArray) ;
        cudaFree (model->RESparamGPU.d_RESmGivenArray) ;

        free (model->RESparamCPU.RESposNodeArray) ;
        cudaFree (model->RESparamGPU.d_RESposNodeArray) ;

        free (model->RESparamCPU.RESnegNodeArray) ;
        cudaFree (model->RESparamGPU.d_RESnegNodeArray) ;
    }

    return (OK) ;
}
