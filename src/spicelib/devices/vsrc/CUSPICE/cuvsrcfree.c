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
#include "vsrcdefs.h"
#include "ngspice/CUSPICE/CUSPICE.h"

int
cuVSRCdestroy
(
GENmodel *inModel
)
{
    VSRCmodel *model = (VSRCmodel *)inModel ;
    VSRCinstance *here ;
    int i ;

    for ( ; model != NULL ; model = VSRCnextModel(model))
    {
        /* Special case VSRCparamGPU.VSRCcoeffsArray */
        i = 0 ;

        for (here = VSRCinstances(model); here != NULL ; here = VSRCnextInstance(here))
        {
            cudaFree (model->VSRCparamCPU.VSRCcoeffsArray [i]) ;

            i++ ;
        }
        free (model->VSRCparamCPU.VSRCcoeffsArray) ;
        cudaFree (model->VSRCparamGPU.d_VSRCcoeffsArray) ;

        i = 0 ;

        for (here = VSRCinstances(model); here != NULL ; here = VSRCnextInstance(here))
        {
            free (model->VSRCparamCPU.VSRCcoeffsArrayHost [i]) ;

            i++ ;
        }
        free (model->VSRCparamCPU.VSRCcoeffsArrayHost) ;
        /* ----------------------------------------- */

        /* DOUBLE */
        free (model->VSRCparamCPU.VSRCdcvalueArray) ;
        cudaFree (model->VSRCparamGPU.d_VSRCdcvalueArray) ;

        free (model->VSRCparamCPU.VSRCrdelayArray) ;
        cudaFree (model->VSRCparamGPU.d_VSRCrdelayArray) ;

        free (model->VSRCparamCPU.VSRCValueArray) ;
        cudaFree (model->VSRCparamGPU.d_VSRCValueArray) ;

        /* INT */
        free (model->VSRCparamCPU.VSRCdcGivenArray) ;
        cudaFree (model->VSRCparamGPU.d_VSRCdcGivenArray) ;

        free (model->VSRCparamCPU.VSRCfunctionTypeArray) ;
        cudaFree (model->VSRCparamGPU.d_VSRCfunctionTypeArray) ;

        free (model->VSRCparamCPU.VSRCfunctionOrderArray) ;
        cudaFree (model->VSRCparamGPU.d_VSRCfunctionOrderArray) ;

        free (model->VSRCparamCPU.VSRCrGivenArray) ;
        cudaFree (model->VSRCparamGPU.d_VSRCrGivenArray) ;

        free (model->VSRCparamCPU.VSRCrBreakptArray) ;
        cudaFree (model->VSRCparamGPU.d_VSRCrBreakptArray) ;
    }

    return (OK) ;
}
