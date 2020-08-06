/*******************************************************************************
 * Copyright 2020 Florian Ballenegger, Anamosic Ballenegger Design
 *******************************************************************************
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

#include "ngspice/ngspice.h"
#include "bsim3def.h"

#define NDATASIMD 7

#ifndef USE_OMP
extern int BSIM3SIMDLoadSeq(BSIM3instance *here, CKTcircuit *ckt, double* data, int stride);
extern int BSIM3LoadSIMD(BSIM3instance **heres, CKTcircuit *ckt, double data[7][NSIMD]);
#else
extern void BSIM3SIMDLoadRhsMat(GENmodel *inModel, CKTcircuit *ckt);
extern int BSIM3SIMDLoadSeq(BSIM3instance *here, CKTcircuit *ckt, int);
extern int BSIM3LoadSIMD(BSIM3instance **heres, CKTcircuit *ckt);
#endif


#ifndef USE_OMP
int
BSIM3SIMDloadSel (GENmodel *inModel, CKTcircuit *ckt)
{
	#ifndef USE_OMP
	double data[NDATASIMD][NSIMD];
	#endif
    BSIM3group *group; /* a group of instance of same model, same pParam, same nqsMode, same geo and same off */
    BSIM3model *model = (BSIM3model*)inModel;
    BSIM3instance* heres[NSIMD];
    
	for (; model != NULL; model = BSIM3SIMDnextModel(model))
	for (group=model->groupHead; group!=NULL; group=group->next)
	{
    	   int idx=0;
    	   while(idx+NSIMD <= group->InstCount)
    	   {
    		int count=0;
    		while((count<NSIMD) && (idx<group->InstCount))
    		{	
			data[0][count]=NAN;
			heres[count] = group->InstArray[idx];
    			int local_error = BSIM3SIMDLoadSeq(group->InstArray[idx++],ckt,
				&data[0][count],NSIMD
			);
			if (local_error) return local_error;
			if(!isnan(data[0][count]))
			{
				count++;
			}
    		}
    		if(count==NSIMD)
    		{
			int local_error;
			 /* process NSIMD instances at once */
			local_error = BSIM3LoadSIMD(heres, ckt, data);
        		if (local_error) return local_error;
    		}
		else for(int i=0;i<count;i++)
		{
			int local_error = BSIM3SIMDLoadSeq(heres[i], ckt, NULL,0);
        		if (local_error) return local_error;
		}
    	   }
	
    	   /* remaining instances are evaluated sequencially */
    	   for (; idx < group->InstCount; idx++) {
    		int local_error = BSIM3SIMDLoadSeq(group->InstArray[idx], ckt,
			NULL, 0);
        	if (local_error) return local_error;
           }
    }

    return 0; /* no error */
}
#endif


#ifdef USE_OMP
int
BSIM3SIMDloadSel (GENmodel *inModel, CKTcircuit *ckt)
{
	/*
	This version do omp parallel only inside groups
	*/
	BSIM3group *group;
	BSIM3model *model = (BSIM3model*)inModel;
	int error=0;
	int idx=0;
	for (; model != NULL; model = BSIM3SIMDnextModel(model))
	for (group=model->groupHead; group!=NULL; group=group->next)
	{	
	
	#pragma omp parallel for
	for (idx=0; idx <= group->InstCount-NSIMD; idx+=NSIMD)
	{
    		int local_error;
		int i;
		int needeval=0;
		for(i=0;i<NSIMD;i++)
		{
			group->InstArray[idx+i]->BSIM3SIMDCheck=-1;
			local_error = BSIM3SIMDLoadSeq(group->InstArray[idx+i], ckt, 1);
        		if (local_error) error = local_error;
			
			if(group->InstArray[idx+i]->BSIM3SIMDCheck!=-1)
				needeval=1;
		}
		if(!needeval)
			continue; /* all NSIMD instances are bypassed */
		local_error = BSIM3LoadSIMD(&group->InstArray[idx], ckt);
		if (local_error) error = local_error;
	}
	/* omp mess with idx val after the for loop above, so we recalc it */
	idx = NSIMD*(group->InstCount/NSIMD);
	for (; idx < group->InstCount; idx++) {
		int local_error = BSIM3SIMDLoadSeq(group->InstArray[idx], ckt, 2);
		if (local_error) error = local_error;
	}
	}
	
	BSIM3SIMDLoadRhsMat(inModel, ckt);
	return error;
}

#endif


