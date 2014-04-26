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

int
cuBSIM4v7destroy
(
GENmodel *inModel
)
{
    BSIM4v7model *model = (BSIM4v7model *)inModel ;
    BSIM4v7instance *here ;

    int i ;

    for ( ; model != NULL ; model = BSIM4v7nextModel(model))
    {
        /* Special case here->d_pParam */
        i = 0 ;

        for (here = BSIM4v7instances(model); here != NULL ; here = BSIM4v7nextInstance(here))
        {
            if (here->pParam != NULL)
                cudaFree (model->pParamHost [i]) ;

            i++ ;
        }

        free (model->pParamHost) ;
        cudaFree (model->d_pParam) ;

        /* DOUBLE */
        free (model->BSIM4v7paramCPU.BSIM4v7gbsRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gbsRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cbsRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cbsRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gbdRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gbdRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cbdRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cbdRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7vonRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7vonRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7vdsatRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7vdsatRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7csubRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7csubRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gdsRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gdsRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gmRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gmRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gmbsRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gmbsRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gcrgRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gcrgRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7IgidlRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7IgidlRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7IgislRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7IgislRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7IgcsRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7IgcsRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7IgcdRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7IgcdRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7IgsRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7IgsRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7IgdRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7IgdRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7IgbRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7IgbRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cdRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cdRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7qinvRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7qinvRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cggbRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cggbRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cgsbRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cgsbRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cgdbRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cgdbRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cdgbRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cdgbRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cdsbRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cdsbRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cddbRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cddbRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cbgbRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cbgbRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cbsbRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cbsbRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cbdbRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cbdbRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7csgbRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7csgbRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cssbRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cssbRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7csdbRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7csdbRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cgbbRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cgbbRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7csbbRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7csbbRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cdbbRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cdbbRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cbbbRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cbbbRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gtauRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gtauRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7qgateRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7qgateRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7qbulkRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7qbulkRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7qdrnRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7qdrnRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7qsrcRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7qsrcRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7capbsRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7capbsRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7capbdRWArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7capbdRWArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7icVDSArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7icVDSArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7icVGSArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7icVGSArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7icVBSArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7icVBSArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7vth0Array) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7vth0Array) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gbbsArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gbbsArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7ggidlbArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7ggidlbArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gbgsArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gbgsArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7ggidlgArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7ggidlgArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gbdsArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gbdsArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7ggidldArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7ggidldArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7ggislsArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7ggislsArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7ggislgArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7ggislgArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7ggislbArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7ggislbArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gIgsgArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gIgsgArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gIgcsgArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gIgcsgArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gIgcsdArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gIgcsdArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gIgcsbArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gIgcsbArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gIgdgArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gIgdgArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gIgcdgArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gIgcdgArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gIgcddArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gIgcddArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gIgcdbArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gIgcdbArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gIgbgArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gIgbgArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gIgbdArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gIgbdArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gIgbbArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gIgbbArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7ggidlsArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7ggidlsArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7ggisldArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7ggisldArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gstotArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gstotArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gstotdArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gstotdArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gstotgArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gstotgArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gstotbArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gstotbArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gdtotArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gdtotArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gdtotdArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gdtotdArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gdtotgArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gdtotgArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gdtotbArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gdtotbArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cgdoArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cgdoArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7qgdoArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7qgdoArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cgsoArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cgsoArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7qgsoArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7qgsoArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7AseffArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7AseffArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7PseffArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7PseffArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7nfArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7nfArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7XExpBVSArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7XExpBVSArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7vjsmFwdArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7vjsmFwdArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7IVjsmFwdArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7IVjsmFwdArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7vjsmRevArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7vjsmRevArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7IVjsmRevArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7IVjsmRevArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SslpRevArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SslpRevArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SslpFwdArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SslpFwdArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7AdeffArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7AdeffArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7PdeffArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7PdeffArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7XExpBVDArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7XExpBVDArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7vjdmFwdArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7vjdmFwdArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7IVjdmFwdArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7IVjdmFwdArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7vjdmRevArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7vjdmRevArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7IVjdmRevArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7IVjdmRevArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DslpRevArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DslpRevArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DslpFwdArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DslpFwdArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SjctTempRevSatCurArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SjctTempRevSatCurArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SswTempRevSatCurArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SswTempRevSatCurArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SswgTempRevSatCurArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SswgTempRevSatCurArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DjctTempRevSatCurArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DjctTempRevSatCurArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DswTempRevSatCurArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DswTempRevSatCurArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DswgTempRevSatCurArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DswgTempRevSatCurArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7vbscArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7vbscArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7thetavthArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7thetavthArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7eta0Array) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7eta0Array) ;

        free (model->BSIM4v7paramCPU.BSIM4v7k2oxArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7k2oxArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7nstarArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7nstarArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7vfbArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7vfbArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7vgs_effArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7vgs_effArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7vgd_effArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7vgd_effArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7dvgs_eff_dvgArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7dvgs_eff_dvgArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7dvgd_eff_dvgArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7dvgd_eff_dvgArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7VgsteffArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7VgsteffArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7grdswArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7grdswArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7AbulkArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7AbulkArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7vtfbphi1Array) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7vtfbphi1Array) ;

        free (model->BSIM4v7paramCPU.BSIM4v7ueffArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7ueffArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7u0tempArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7u0tempArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7vsattempArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7vsattempArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7EsatLArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7EsatLArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7VdseffArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7VdseffArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7vtfbphi2Array) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7vtfbphi2Array) ;

        free (model->BSIM4v7paramCPU.BSIM4v7CoxeffArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7CoxeffArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7AbovVgst2VtmArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7AbovVgst2VtmArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7IdovVdsArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7IdovVdsArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gcrgdArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gcrgdArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gcrgbArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gcrgbArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gcrggArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gcrggArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7grgeltdArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7grgeltdArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gcrgsArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gcrgsArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7sourceConductanceArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7sourceConductanceArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7drainConductanceArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7drainConductanceArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gstotsArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gstotsArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gdtotsArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gdtotsArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7vfbzbArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7vfbzbArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gIgssArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gIgssArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gIgddArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gIgddArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gIgbsArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gIgbsArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gIgcssArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gIgcssArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gIgcdsArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gIgcdsArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7noiGd0Array) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7noiGd0Array) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cqdbArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cqdbArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cqsbArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cqsbArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cqgbArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cqgbArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7qchqsArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7qchqsArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7cqbbArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7cqbbArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7taunetArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7taunetArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gtgArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gtgArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gtdArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gtdArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gtsArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gtsArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gtbArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gtbArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7mArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7mArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7grbpdArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7grbpdArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7grbdbArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7grbdbArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7grbpbArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7grbpbArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7grbpsArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7grbpsArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7grbsbArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7grbsbArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7dNodePrimeRHSValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7dNodePrimeRHSValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gNodePrimeRHSValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gNodePrimeRHSValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gNodeExtRHSValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gNodeExtRHSValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gNodeMidRHSValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gNodeMidRHSValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7bNodePrimeRHSValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7bNodePrimeRHSValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7sNodePrimeRHSValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7sNodePrimeRHSValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7dbNodeRHSValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7dbNodeRHSValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7sbNodeRHSValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7sbNodeRHSValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7dNodeRHSValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7dNodeRHSValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7sNodeRHSValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7sNodeRHSValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7qNodeRHSValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7qNodeRHSValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7GEgeValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7GEgeValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7GPgeValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7GPgeValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7GEgpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7GEgpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7GPgpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7GPgpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7GPdpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7GPdpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7GPspValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7GPspValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7GPbpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7GPbpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7GEdpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7GEdpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7GEspValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7GEspValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7GEbpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7GEbpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7GEgmValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7GEgmValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7GMgeValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7GMgeValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7GMgmValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7GMgmValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7GMdpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7GMdpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7GMgpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7GMgpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7GMspValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7GMspValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7GMbpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7GMbpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DPgmValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DPgmValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7GPgmValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7GPgmValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SPgmValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SPgmValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7BPgmValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7BPgmValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DgpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DgpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DspValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DspValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DbpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DbpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SdpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SdpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SgpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SgpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SbpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SbpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DPdpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DPdpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DPdValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DPdValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DPgpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DPgpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DPspValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DPspValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DPbpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DPbpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DdpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DdpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DdValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DdValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SPdpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SPdpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SPgpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SPgpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SPspValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SPspValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SPsValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SPsValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SPbpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SPbpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SspValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SspValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SsValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SsValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7BPdpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7BPdpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7BPgpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7BPgpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7BPspValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7BPspValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7BPbpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7BPbpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DPdbValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DPdbValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SPsbValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SPsbValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DBdpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DBdpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DBdbValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DBdbValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DBbpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DBbpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DBbValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DBbValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7BPdbValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7BPdbValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7BPbValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7BPbValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7BPsbValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7BPsbValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7BPbpIFValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7BPbpIFValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SBspValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SBspValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SBbpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SBbpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SBbValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SBbValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SBsbValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SBsbValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7BdbValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7BdbValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7BbpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7BbpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7BsbValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7BsbValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7BbValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7BbValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7QqValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7QqValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7QgpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7QgpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7QdpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7QdpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7QspValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7QspValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7QbpValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7QbpValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7DPqValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7DPqValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7SPqValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7SPqValueArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7GPqValueArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7GPqValueArray) ;

        /* INT */
        free (model->BSIM4v7paramCPU.BSIM4v7offArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7offArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7dNodePrimeArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7dNodePrimeArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7sNodePrimeArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7sNodePrimeArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gNodePrimeArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gNodePrimeArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7bNodePrimeArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7bNodePrimeArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gNodeExtArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gNodeExtArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7gNodeMidArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7gNodeMidArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7dbNodeArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7dbNodeArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7sbNodeArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7sbNodeArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7sNodeArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7sNodeArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7dNodeArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7dNodeArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7qNodeArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7qNodeArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7rbodyModArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7rbodyModArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7modeArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7modeArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7rgateModArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7rgateModArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7trnqsModArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7trnqsModArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7acnqsModArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7acnqsModArray) ;

        free (model->BSIM4v7paramCPU.BSIM4v7statesArray) ;
        cudaFree (model->BSIM4v7paramGPU.d_BSIM4v7statesArray) ;
    }

    return (OK) ;
}
