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
#include "bsim3v32def.h"

/* NSIMD = number of // instances evaluated (=4 for AVX2) */
#define NDATASIMD 7

#define DEBUG 0

#ifndef USE_OMP
extern int BSIM3v32LoadSeq(BSIM3v32instance *here, CKTcircuit *ckt, double* data, int stride);
extern int BSIM3v32LoadSIMD(BSIM3v32instance **heres, CKTcircuit *ckt, double data[7][NSIMD]);
#else
extern void BSIM3v32LoadRhsMat(GENmodel *inModel, CKTcircuit *ckt);
extern int BSIM3v32LoadSeq(BSIM3v32instance *here, CKTcircuit *ckt, int);
extern int BSIM3v32LoadSIMD(BSIM3v32instance **heres, CKTcircuit *ckt);
#endif


#ifndef USE_OMP
int
BSIM3v32loadSel (GENmodel *inModel, CKTcircuit *ckt)
{
	#ifndef USE_OMP
	double data[NDATASIMD][NSIMD];
	#endif
    BSIM3v32group *group; /* a group of instance of same model, same pParam, same nqsMode, same geo and same off */
    BSIM3v32model *model = (BSIM3v32model*)inModel;
    BSIM3v32instance* heres[NSIMD];
    
	for (; model != NULL; model = BSIM3v32nextModel(model))
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
    			int local_error = BSIM3v32LoadSeq(group->InstArray[idx++],ckt,
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
			local_error = BSIM3v32LoadSIMD(heres, ckt, data);
        		if (local_error) return local_error;
    		}
		else for(int i=0;i<count;i++)
		{
			int local_error = BSIM3v32LoadSeq(heres[i], ckt, NULL,0);
        		if (local_error) return local_error;
		}
    	   }
	
    	   /* remaining instances are evaluated sequencially */
    	   for (; idx < group->InstCount; idx++) {
    		int local_error = BSIM3v32LoadSeq(group->InstArray[idx], ckt,
			NULL, 0);
        	if (local_error) return local_error;
           }
    }

    return 0; /* no error */
}
#endif


#ifdef USE_OMP
int
BSIM3v32loadSel (GENmodel *inModel, CKTcircuit *ckt)
{
	/*
	This version do omp parallel only inside groups
	*/
	BSIM3v32group *group;
	BSIM3v32model *model = (BSIM3v32model*)inModel;
	int error=0;
	int idx=0;
	for (; model != NULL; model = BSIM3v32nextModel(model))
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
			group->InstArray[idx+i]->BSIM3v32SIMDCheck=-1;
			local_error = BSIM3v32LoadSeq(group->InstArray[idx+i], ckt, 1);
        		if (local_error) error = local_error;
			
			if(group->InstArray[idx+i]->BSIM3v32SIMDCheck!=-1)
				needeval=1;
		}
		if(!needeval)
			continue; /* all NSIMD instances are bypassed */
		local_error = BSIM3v32LoadSIMD(&group->InstArray[idx], ckt);
		if (local_error) error = local_error;
	}
	/* omp mess with idx val after the for loop above, so we recalc it */
	idx = NSIMD*(group->InstCount/NSIMD);
	for (; idx < group->InstCount; idx++) {
		int local_error = BSIM3v32LoadSeq(group->InstArray[idx], ckt, 2);
		if (local_error) error = local_error;
	}
	}
	
	BSIM3v32LoadRhsMat(inModel, ckt);
	return error;
}

#if 0
int
BSIM3v32loadSelVrai (GENmodel *inModel, CKTcircuit *ckt)
{
	/*
	This version do omp parallel for most instances of all models combined 
	*/
	BSIM3v32group *group;
	BSIM3v32model *model = (BSIM3v32model*)inModel;
    
    	int idx;
	int error = 0;
	int nsimd,nsisd;
	/* pre load all instances */
	if(DEBUG) printf("loadomp %d\n",model->BSIM3v32InstCount);
	#pragma omp parallel for
	for (idx = 0; idx < model->BSIM3v32InstCount; idx++) {
		BSIM3v32instance *here = model->BSIM3v32InstanceArray[idx];
		if(DEBUG) printf("loadomp preload seq %d\n",idx);
		here->BSIM3v32SIMDCheck=-1;
		int local_error = BSIM3v32LoadSeq(here,ckt,1);
		if (local_error) error=local_error;
	}
	if (error) printf("load error\n");
        if (error) return error;
	
	/* sort instances to run in SIMD */
	nsimd=0;
	nsisd=0;
	for (model = (BSIM3v32model*)inModel; model != NULL; model = BSIM3v32nextModel(model))
	for (group=model->groupHead; group!=NULL; group=group->next)
	{
		int rev=group->InstCount;
		group->SimdCount = 0;
		for(idx=0;idx<group->InstCount;idx++)
		{
			BSIM3v32instance *here = group->InstArray[idx];
			if(here->BSIM3v32SIMDCheck==-1)
			{
				/* bypassed, swap current inst to the end */
				rev--;
				group->InstArray[idx] = group->InstArray[rev];
				group->InstArray[rev] = here;
			}
		}
		group->EvalCount = rev;
		group->SimdCount = rev/NSIMD;
		nsimd += group->SimdCount;
		nsisd += rev - NSIMD*group->SimdCount;
	}
	
	if(DEBUG) printf("nsimd=%d nsisd=%d\n",nsimd,nsisd);
	/* run SIMD in parallel */
	#pragma omp parallel for
	for(idx=0;idx<nsimd;idx++)
	{
		if(DEBUG) printf("Search SIMD index %d\n", idx);
		int search=idx;
		BSIM3v32model* mod;
		BSIM3v32group* grp;
		for (mod = (BSIM3v32model*)inModel; mod != NULL; mod = BSIM3v32nextModel(mod))
		for (grp=mod->groupHead; grp!=NULL; grp=grp->next)
		{
			if(search>=0 && search < grp->SimdCount)
			{
				BSIM3v32instance** heres = &grp->InstArray[search*NSIMD];
				if(DEBUG) printf("Call Simd index %d of %d\n", search*NSIMD, grp->InstCount);
				int local_error = BSIM3v32LoadSIMD(heres, ckt);
				if(DEBUG) printf("Call ended\n");
				if(local_error) error=local_error;
			}
			search -= grp->SimdCount;
		}
	}
	if(error) return error;
	
	if(DEBUG) printf("now switch to sisd\n");
	/* run remaining SISD in parallel */
	#pragma omp parallel for
	for(idx=0;idx<nsisd;idx++)
	{
		int search=idx;
		BSIM3v32model* mod;
		BSIM3v32group* grp;
		for (mod = (BSIM3v32model*)inModel; mod != NULL; mod = BSIM3v32nextModel(mod))
		for (grp=mod->groupHead; grp!=NULL; grp=grp->next)
		{
			int n = grp->EvalCount - grp->SimdCount*NSIMD;
			if(search>=0 && search < n)
			{
				if(DEBUG) printf("Call seq index %d of %d\n", search + grp->SimdCount*NSIMD,grp->InstCount);
				int local_error = BSIM3v32LoadSeq(grp->InstArray[search + grp->SimdCount*NSIMD], ckt, 0);
				if(DEBUG) printf("Call ended\n");
				if(local_error) error=local_error;
			}
			search -= n;
		}
	}
	if(DEBUG) printf("Now write the matrix\n");
	/* Write in matrix sequentially */
	BSIM3v32LoadRhsMat(inModel, ckt);
	
	return error;
}
#endif

#endif


