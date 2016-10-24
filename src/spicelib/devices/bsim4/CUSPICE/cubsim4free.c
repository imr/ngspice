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

int
cuBSIM4destroy
(
GENmodel *inModel
)
{
    BSIM4model *model = (BSIM4model *)inModel ;
    BSIM4instance *here ;

    int i ;

    for ( ; model != NULL ; model = model->BSIM4nextModel)
    {
        /* Special case here->d_pParam */
        i = 0 ;

        for (here = model->BSIM4instances ; here != NULL ; here = here->BSIM4nextInstance)
        {
            if (here->pParam != NULL)
                cudaFree (model->pParamHost [i]) ;

            i++ ;
        }

        free (model->pParamHost) ;
        cudaFree (model->d_pParam) ;

        /* DOUBLE */
        free (model->BSIM4paramCPU.BSIM4gbsRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gbsRWArray) ;

        free (model->BSIM4paramCPU.BSIM4cbsRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cbsRWArray) ;

        free (model->BSIM4paramCPU.BSIM4gbdRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gbdRWArray) ;

        free (model->BSIM4paramCPU.BSIM4cbdRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cbdRWArray) ;

        free (model->BSIM4paramCPU.BSIM4vonRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4vonRWArray) ;

        free (model->BSIM4paramCPU.BSIM4vdsatRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4vdsatRWArray) ;

        free (model->BSIM4paramCPU.BSIM4csubRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4csubRWArray) ;

        free (model->BSIM4paramCPU.BSIM4gdsRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gdsRWArray) ;

        free (model->BSIM4paramCPU.BSIM4gmRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gmRWArray) ;

        free (model->BSIM4paramCPU.BSIM4gmbsRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gmbsRWArray) ;

        free (model->BSIM4paramCPU.BSIM4gcrgRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gcrgRWArray) ;

        free (model->BSIM4paramCPU.BSIM4IgidlRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4IgidlRWArray) ;

        free (model->BSIM4paramCPU.BSIM4IgislRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4IgislRWArray) ;

        free (model->BSIM4paramCPU.BSIM4IgcsRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4IgcsRWArray) ;

        free (model->BSIM4paramCPU.BSIM4IgcdRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4IgcdRWArray) ;

        free (model->BSIM4paramCPU.BSIM4IgsRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4IgsRWArray) ;

        free (model->BSIM4paramCPU.BSIM4IgdRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4IgdRWArray) ;

        free (model->BSIM4paramCPU.BSIM4IgbRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4IgbRWArray) ;

        free (model->BSIM4paramCPU.BSIM4cdRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cdRWArray) ;

        free (model->BSIM4paramCPU.BSIM4qinvRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4qinvRWArray) ;

        free (model->BSIM4paramCPU.BSIM4cggbRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cggbRWArray) ;

        free (model->BSIM4paramCPU.BSIM4cgsbRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cgsbRWArray) ;

        free (model->BSIM4paramCPU.BSIM4cgdbRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cgdbRWArray) ;

        free (model->BSIM4paramCPU.BSIM4cdgbRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cdgbRWArray) ;

        free (model->BSIM4paramCPU.BSIM4cdsbRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cdsbRWArray) ;

        free (model->BSIM4paramCPU.BSIM4cddbRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cddbRWArray) ;

        free (model->BSIM4paramCPU.BSIM4cbgbRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cbgbRWArray) ;

        free (model->BSIM4paramCPU.BSIM4cbsbRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cbsbRWArray) ;

        free (model->BSIM4paramCPU.BSIM4cbdbRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cbdbRWArray) ;

        free (model->BSIM4paramCPU.BSIM4csgbRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4csgbRWArray) ;

        free (model->BSIM4paramCPU.BSIM4cssbRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cssbRWArray) ;

        free (model->BSIM4paramCPU.BSIM4csdbRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4csdbRWArray) ;

        free (model->BSIM4paramCPU.BSIM4cgbbRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cgbbRWArray) ;

        free (model->BSIM4paramCPU.BSIM4csbbRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4csbbRWArray) ;

        free (model->BSIM4paramCPU.BSIM4cdbbRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cdbbRWArray) ;

        free (model->BSIM4paramCPU.BSIM4cbbbRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cbbbRWArray) ;

        free (model->BSIM4paramCPU.BSIM4gtauRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gtauRWArray) ;

        free (model->BSIM4paramCPU.BSIM4qgateRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4qgateRWArray) ;

        free (model->BSIM4paramCPU.BSIM4qbulkRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4qbulkRWArray) ;

        free (model->BSIM4paramCPU.BSIM4qdrnRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4qdrnRWArray) ;

        free (model->BSIM4paramCPU.BSIM4qsrcRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4qsrcRWArray) ;

        free (model->BSIM4paramCPU.BSIM4capbsRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4capbsRWArray) ;

        free (model->BSIM4paramCPU.BSIM4capbdRWArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4capbdRWArray) ;

        free (model->BSIM4paramCPU.BSIM4icVDSArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4icVDSArray) ;

        free (model->BSIM4paramCPU.BSIM4icVGSArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4icVGSArray) ;

        free (model->BSIM4paramCPU.BSIM4icVBSArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4icVBSArray) ;

        free (model->BSIM4paramCPU.BSIM4vth0Array) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4vth0Array) ;

        free (model->BSIM4paramCPU.BSIM4gbbsArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gbbsArray) ;

        free (model->BSIM4paramCPU.BSIM4ggidlbArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4ggidlbArray) ;

        free (model->BSIM4paramCPU.BSIM4gbgsArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gbgsArray) ;

        free (model->BSIM4paramCPU.BSIM4ggidlgArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4ggidlgArray) ;

        free (model->BSIM4paramCPU.BSIM4gbdsArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gbdsArray) ;

        free (model->BSIM4paramCPU.BSIM4ggidldArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4ggidldArray) ;

        free (model->BSIM4paramCPU.BSIM4ggislsArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4ggislsArray) ;

        free (model->BSIM4paramCPU.BSIM4ggislgArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4ggislgArray) ;

        free (model->BSIM4paramCPU.BSIM4ggislbArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4ggislbArray) ;

        free (model->BSIM4paramCPU.BSIM4gIgsgArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gIgsgArray) ;

        free (model->BSIM4paramCPU.BSIM4gIgcsgArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gIgcsgArray) ;

        free (model->BSIM4paramCPU.BSIM4gIgcsdArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gIgcsdArray) ;

        free (model->BSIM4paramCPU.BSIM4gIgcsbArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gIgcsbArray) ;

        free (model->BSIM4paramCPU.BSIM4gIgdgArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gIgdgArray) ;

        free (model->BSIM4paramCPU.BSIM4gIgcdgArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gIgcdgArray) ;

        free (model->BSIM4paramCPU.BSIM4gIgcddArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gIgcddArray) ;

        free (model->BSIM4paramCPU.BSIM4gIgcdbArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gIgcdbArray) ;

        free (model->BSIM4paramCPU.BSIM4gIgbgArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gIgbgArray) ;

        free (model->BSIM4paramCPU.BSIM4gIgbdArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gIgbdArray) ;

        free (model->BSIM4paramCPU.BSIM4gIgbbArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gIgbbArray) ;

        free (model->BSIM4paramCPU.BSIM4ggidlsArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4ggidlsArray) ;

        free (model->BSIM4paramCPU.BSIM4ggisldArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4ggisldArray) ;

        free (model->BSIM4paramCPU.BSIM4gstotArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gstotArray) ;

        free (model->BSIM4paramCPU.BSIM4gstotdArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gstotdArray) ;

        free (model->BSIM4paramCPU.BSIM4gstotgArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gstotgArray) ;

        free (model->BSIM4paramCPU.BSIM4gstotbArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gstotbArray) ;

        free (model->BSIM4paramCPU.BSIM4gdtotArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gdtotArray) ;

        free (model->BSIM4paramCPU.BSIM4gdtotdArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gdtotdArray) ;

        free (model->BSIM4paramCPU.BSIM4gdtotgArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gdtotgArray) ;

        free (model->BSIM4paramCPU.BSIM4gdtotbArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gdtotbArray) ;

        free (model->BSIM4paramCPU.BSIM4cgdoArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cgdoArray) ;

        free (model->BSIM4paramCPU.BSIM4qgdoArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4qgdoArray) ;

        free (model->BSIM4paramCPU.BSIM4cgsoArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cgsoArray) ;

        free (model->BSIM4paramCPU.BSIM4qgsoArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4qgsoArray) ;

        free (model->BSIM4paramCPU.BSIM4AseffArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4AseffArray) ;

        free (model->BSIM4paramCPU.BSIM4PseffArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4PseffArray) ;

        free (model->BSIM4paramCPU.BSIM4nfArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4nfArray) ;

        free (model->BSIM4paramCPU.BSIM4XExpBVSArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4XExpBVSArray) ;

        free (model->BSIM4paramCPU.BSIM4vjsmFwdArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4vjsmFwdArray) ;

        free (model->BSIM4paramCPU.BSIM4IVjsmFwdArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4IVjsmFwdArray) ;

        free (model->BSIM4paramCPU.BSIM4vjsmRevArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4vjsmRevArray) ;

        free (model->BSIM4paramCPU.BSIM4IVjsmRevArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4IVjsmRevArray) ;

        free (model->BSIM4paramCPU.BSIM4SslpRevArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SslpRevArray) ;

        free (model->BSIM4paramCPU.BSIM4SslpFwdArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SslpFwdArray) ;

        free (model->BSIM4paramCPU.BSIM4AdeffArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4AdeffArray) ;

        free (model->BSIM4paramCPU.BSIM4PdeffArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4PdeffArray) ;

        free (model->BSIM4paramCPU.BSIM4XExpBVDArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4XExpBVDArray) ;

        free (model->BSIM4paramCPU.BSIM4vjdmFwdArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4vjdmFwdArray) ;

        free (model->BSIM4paramCPU.BSIM4IVjdmFwdArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4IVjdmFwdArray) ;

        free (model->BSIM4paramCPU.BSIM4vjdmRevArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4vjdmRevArray) ;

        free (model->BSIM4paramCPU.BSIM4IVjdmRevArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4IVjdmRevArray) ;

        free (model->BSIM4paramCPU.BSIM4DslpRevArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DslpRevArray) ;

        free (model->BSIM4paramCPU.BSIM4DslpFwdArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DslpFwdArray) ;

        free (model->BSIM4paramCPU.BSIM4SjctTempRevSatCurArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SjctTempRevSatCurArray) ;

        free (model->BSIM4paramCPU.BSIM4SswTempRevSatCurArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SswTempRevSatCurArray) ;

        free (model->BSIM4paramCPU.BSIM4SswgTempRevSatCurArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SswgTempRevSatCurArray) ;

        free (model->BSIM4paramCPU.BSIM4DjctTempRevSatCurArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DjctTempRevSatCurArray) ;

        free (model->BSIM4paramCPU.BSIM4DswTempRevSatCurArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DswTempRevSatCurArray) ;

        free (model->BSIM4paramCPU.BSIM4DswgTempRevSatCurArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DswgTempRevSatCurArray) ;

        free (model->BSIM4paramCPU.BSIM4vbscArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4vbscArray) ;

        free (model->BSIM4paramCPU.BSIM4thetavthArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4thetavthArray) ;

        free (model->BSIM4paramCPU.BSIM4eta0Array) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4eta0Array) ;

        free (model->BSIM4paramCPU.BSIM4k2oxArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4k2oxArray) ;

        free (model->BSIM4paramCPU.BSIM4nstarArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4nstarArray) ;

        free (model->BSIM4paramCPU.BSIM4vfbArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4vfbArray) ;

        free (model->BSIM4paramCPU.BSIM4vgs_effArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4vgs_effArray) ;

        free (model->BSIM4paramCPU.BSIM4vgd_effArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4vgd_effArray) ;

        free (model->BSIM4paramCPU.BSIM4dvgs_eff_dvgArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4dvgs_eff_dvgArray) ;

        free (model->BSIM4paramCPU.BSIM4dvgd_eff_dvgArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4dvgd_eff_dvgArray) ;

        free (model->BSIM4paramCPU.BSIM4VgsteffArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4VgsteffArray) ;

        free (model->BSIM4paramCPU.BSIM4grdswArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4grdswArray) ;

        free (model->BSIM4paramCPU.BSIM4AbulkArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4AbulkArray) ;

        free (model->BSIM4paramCPU.BSIM4vtfbphi1Array) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4vtfbphi1Array) ;

        free (model->BSIM4paramCPU.BSIM4ueffArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4ueffArray) ;

        free (model->BSIM4paramCPU.BSIM4u0tempArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4u0tempArray) ;

        free (model->BSIM4paramCPU.BSIM4vsattempArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4vsattempArray) ;

        free (model->BSIM4paramCPU.BSIM4EsatLArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4EsatLArray) ;

        free (model->BSIM4paramCPU.BSIM4VdseffArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4VdseffArray) ;

        free (model->BSIM4paramCPU.BSIM4vtfbphi2Array) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4vtfbphi2Array) ;

        free (model->BSIM4paramCPU.BSIM4CoxeffArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4CoxeffArray) ;

        free (model->BSIM4paramCPU.BSIM4AbovVgst2VtmArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4AbovVgst2VtmArray) ;

        free (model->BSIM4paramCPU.BSIM4IdovVdsArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4IdovVdsArray) ;

        free (model->BSIM4paramCPU.BSIM4gcrgdArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gcrgdArray) ;

        free (model->BSIM4paramCPU.BSIM4gcrgbArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gcrgbArray) ;

        free (model->BSIM4paramCPU.BSIM4gcrggArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gcrggArray) ;

        free (model->BSIM4paramCPU.BSIM4grgeltdArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4grgeltdArray) ;

        free (model->BSIM4paramCPU.BSIM4gcrgsArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gcrgsArray) ;

        free (model->BSIM4paramCPU.BSIM4sourceConductanceArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4sourceConductanceArray) ;

        free (model->BSIM4paramCPU.BSIM4drainConductanceArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4drainConductanceArray) ;

        free (model->BSIM4paramCPU.BSIM4gstotsArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gstotsArray) ;

        free (model->BSIM4paramCPU.BSIM4gdtotsArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gdtotsArray) ;

        free (model->BSIM4paramCPU.BSIM4vfbzbArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4vfbzbArray) ;

        free (model->BSIM4paramCPU.BSIM4gIgssArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gIgssArray) ;

        free (model->BSIM4paramCPU.BSIM4gIgddArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gIgddArray) ;

        free (model->BSIM4paramCPU.BSIM4gIgbsArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gIgbsArray) ;

        free (model->BSIM4paramCPU.BSIM4gIgcssArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gIgcssArray) ;

        free (model->BSIM4paramCPU.BSIM4gIgcdsArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gIgcdsArray) ;

        free (model->BSIM4paramCPU.BSIM4noiGd0Array) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4noiGd0Array) ;

        free (model->BSIM4paramCPU.BSIM4cqdbArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cqdbArray) ;

        free (model->BSIM4paramCPU.BSIM4cqsbArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cqsbArray) ;

        free (model->BSIM4paramCPU.BSIM4cqgbArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cqgbArray) ;

        free (model->BSIM4paramCPU.BSIM4qchqsArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4qchqsArray) ;

        free (model->BSIM4paramCPU.BSIM4cqbbArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4cqbbArray) ;

        free (model->BSIM4paramCPU.BSIM4taunetArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4taunetArray) ;

        free (model->BSIM4paramCPU.BSIM4gtgArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gtgArray) ;

        free (model->BSIM4paramCPU.BSIM4gtdArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gtdArray) ;

        free (model->BSIM4paramCPU.BSIM4gtsArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gtsArray) ;

        free (model->BSIM4paramCPU.BSIM4gtbArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gtbArray) ;

        free (model->BSIM4paramCPU.BSIM4mArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4mArray) ;

        free (model->BSIM4paramCPU.BSIM4grbpdArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4grbpdArray) ;

        free (model->BSIM4paramCPU.BSIM4grbdbArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4grbdbArray) ;

        free (model->BSIM4paramCPU.BSIM4grbpbArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4grbpbArray) ;

        free (model->BSIM4paramCPU.BSIM4grbpsArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4grbpsArray) ;

        free (model->BSIM4paramCPU.BSIM4grbsbArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4grbsbArray) ;

        free (model->BSIM4paramCPU.BSIM4dNodePrimeRHSValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4dNodePrimeRHSValueArray) ;

        free (model->BSIM4paramCPU.BSIM4gNodePrimeRHSValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gNodePrimeRHSValueArray) ;

        free (model->BSIM4paramCPU.BSIM4gNodeExtRHSValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gNodeExtRHSValueArray) ;

        free (model->BSIM4paramCPU.BSIM4gNodeMidRHSValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gNodeMidRHSValueArray) ;

        free (model->BSIM4paramCPU.BSIM4bNodePrimeRHSValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4bNodePrimeRHSValueArray) ;

        free (model->BSIM4paramCPU.BSIM4sNodePrimeRHSValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4sNodePrimeRHSValueArray) ;

        free (model->BSIM4paramCPU.BSIM4dbNodeRHSValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4dbNodeRHSValueArray) ;

        free (model->BSIM4paramCPU.BSIM4sbNodeRHSValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4sbNodeRHSValueArray) ;

        free (model->BSIM4paramCPU.BSIM4dNodeRHSValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4dNodeRHSValueArray) ;

        free (model->BSIM4paramCPU.BSIM4sNodeRHSValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4sNodeRHSValueArray) ;

        free (model->BSIM4paramCPU.BSIM4qNodeRHSValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4qNodeRHSValueArray) ;

        free (model->BSIM4paramCPU.BSIM4GEgeValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4GEgeValueArray) ;

        free (model->BSIM4paramCPU.BSIM4GPgeValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4GPgeValueArray) ;

        free (model->BSIM4paramCPU.BSIM4GEgpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4GEgpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4GPgpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4GPgpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4GPdpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4GPdpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4GPspValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4GPspValueArray) ;

        free (model->BSIM4paramCPU.BSIM4GPbpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4GPbpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4GEdpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4GEdpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4GEspValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4GEspValueArray) ;

        free (model->BSIM4paramCPU.BSIM4GEbpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4GEbpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4GEgmValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4GEgmValueArray) ;

        free (model->BSIM4paramCPU.BSIM4GMgeValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4GMgeValueArray) ;

        free (model->BSIM4paramCPU.BSIM4GMgmValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4GMgmValueArray) ;

        free (model->BSIM4paramCPU.BSIM4GMdpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4GMdpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4GMgpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4GMgpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4GMspValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4GMspValueArray) ;

        free (model->BSIM4paramCPU.BSIM4GMbpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4GMbpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4DPgmValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DPgmValueArray) ;

        free (model->BSIM4paramCPU.BSIM4GPgmValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4GPgmValueArray) ;

        free (model->BSIM4paramCPU.BSIM4SPgmValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SPgmValueArray) ;

        free (model->BSIM4paramCPU.BSIM4BPgmValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4BPgmValueArray) ;

        free (model->BSIM4paramCPU.BSIM4DgpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DgpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4DspValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DspValueArray) ;

        free (model->BSIM4paramCPU.BSIM4DbpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DbpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4SdpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SdpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4SgpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SgpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4SbpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SbpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4DPdpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DPdpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4DPdValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DPdValueArray) ;

        free (model->BSIM4paramCPU.BSIM4DPgpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DPgpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4DPspValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DPspValueArray) ;

        free (model->BSIM4paramCPU.BSIM4DPbpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DPbpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4DdpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DdpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4DdValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DdValueArray) ;

        free (model->BSIM4paramCPU.BSIM4SPdpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SPdpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4SPgpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SPgpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4SPspValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SPspValueArray) ;

        free (model->BSIM4paramCPU.BSIM4SPsValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SPsValueArray) ;

        free (model->BSIM4paramCPU.BSIM4SPbpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SPbpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4SspValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SspValueArray) ;

        free (model->BSIM4paramCPU.BSIM4SsValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SsValueArray) ;

        free (model->BSIM4paramCPU.BSIM4BPdpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4BPdpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4BPgpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4BPgpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4BPspValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4BPspValueArray) ;

        free (model->BSIM4paramCPU.BSIM4BPbpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4BPbpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4DPdbValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DPdbValueArray) ;

        free (model->BSIM4paramCPU.BSIM4SPsbValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SPsbValueArray) ;

        free (model->BSIM4paramCPU.BSIM4DBdpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DBdpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4DBdbValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DBdbValueArray) ;

        free (model->BSIM4paramCPU.BSIM4DBbpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DBbpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4DBbValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DBbValueArray) ;

        free (model->BSIM4paramCPU.BSIM4BPdbValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4BPdbValueArray) ;

        free (model->BSIM4paramCPU.BSIM4BPbValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4BPbValueArray) ;

        free (model->BSIM4paramCPU.BSIM4BPsbValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4BPsbValueArray) ;

        free (model->BSIM4paramCPU.BSIM4BPbpIFValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4BPbpIFValueArray) ;

        free (model->BSIM4paramCPU.BSIM4SBspValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SBspValueArray) ;

        free (model->BSIM4paramCPU.BSIM4SBbpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SBbpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4SBbValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SBbValueArray) ;

        free (model->BSIM4paramCPU.BSIM4SBsbValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SBsbValueArray) ;

        free (model->BSIM4paramCPU.BSIM4BdbValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4BdbValueArray) ;

        free (model->BSIM4paramCPU.BSIM4BbpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4BbpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4BsbValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4BsbValueArray) ;

        free (model->BSIM4paramCPU.BSIM4BbValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4BbValueArray) ;

        free (model->BSIM4paramCPU.BSIM4QqValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4QqValueArray) ;

        free (model->BSIM4paramCPU.BSIM4QgpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4QgpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4QdpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4QdpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4QspValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4QspValueArray) ;

        free (model->BSIM4paramCPU.BSIM4QbpValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4QbpValueArray) ;

        free (model->BSIM4paramCPU.BSIM4DPqValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4DPqValueArray) ;

        free (model->BSIM4paramCPU.BSIM4SPqValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4SPqValueArray) ;

        free (model->BSIM4paramCPU.BSIM4GPqValueArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4GPqValueArray) ;

        /* INT */
        free (model->BSIM4paramCPU.BSIM4offArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4offArray) ;

        free (model->BSIM4paramCPU.BSIM4dNodePrimeArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4dNodePrimeArray) ;

        free (model->BSIM4paramCPU.BSIM4sNodePrimeArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4sNodePrimeArray) ;

        free (model->BSIM4paramCPU.BSIM4gNodePrimeArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gNodePrimeArray) ;

        free (model->BSIM4paramCPU.BSIM4bNodePrimeArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4bNodePrimeArray) ;

        free (model->BSIM4paramCPU.BSIM4gNodeExtArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gNodeExtArray) ;

        free (model->BSIM4paramCPU.BSIM4gNodeMidArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4gNodeMidArray) ;

        free (model->BSIM4paramCPU.BSIM4dbNodeArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4dbNodeArray) ;

        free (model->BSIM4paramCPU.BSIM4sbNodeArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4sbNodeArray) ;

        free (model->BSIM4paramCPU.BSIM4sNodeArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4sNodeArray) ;

        free (model->BSIM4paramCPU.BSIM4dNodeArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4dNodeArray) ;

        free (model->BSIM4paramCPU.BSIM4qNodeArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4qNodeArray) ;

        free (model->BSIM4paramCPU.BSIM4rbodyModArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4rbodyModArray) ;

        free (model->BSIM4paramCPU.BSIM4modeArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4modeArray) ;

        free (model->BSIM4paramCPU.BSIM4rgateModArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4rgateModArray) ;

        free (model->BSIM4paramCPU.BSIM4trnqsModArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4trnqsModArray) ;

        free (model->BSIM4paramCPU.BSIM4acnqsModArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4acnqsModArray) ;

        free (model->BSIM4paramCPU.BSIM4statesArray) ;
        cudaFree (model->BSIM4paramGPU.d_BSIM4statesArray) ;
    }

    return (OK) ;
}
