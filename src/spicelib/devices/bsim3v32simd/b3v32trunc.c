/**** BSIM3v3.2.4, Released by Xuemei Xi 12/21/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b3trunc.c of BSIM3v3.2.4
 * Author: 1995 Min-Chie Jeng and Mansun Chan.
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 * Modified by Poalo Nenzi 2002
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3v32def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/SIMD/simdvector.h"
#include "ngspice/SIMD/simdckt.h"
#include <float.h>

int
BSIM3v32SIMDtruncSeq (GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
BSIM3v32model *model = (BSIM3v32model*)inModel;
BSIM3v32instance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = BSIM3v32nextModel(model))
    {    for (here = BSIM3v32instances(model); here != NULL;
              here = BSIM3v32nextInstance(here))
         {
#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->BSIM3v32qb,ckt,timeStep);
            CKTterr(here->BSIM3v32qg,ckt,timeStep);
            CKTterr(here->BSIM3v32qd,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
            {  printf("device %s reduces step from %g to %g\n",
                       here->BSIM3v32name,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}

/* omp multiprocessing is found to be counter-productive here, probably due to
overhead, so we disable it for the BSIM3v32SIMDtrunc function */
#undef USE_OMP

int
BSIM3v32SIMDtrunc (GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
BSIM3v32model *model = (BSIM3v32model*)inModel;
BSIM3v32instance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */
    int i;
    #ifndef USE_OMP
    #define TIMESTEPptr timeStep
    #else
    double reduTimeStep = *timeStep;
    #define TIMESTEPptr &locTimeStep
    #pragma omp parallel for reduction(min:reduTimeStep)
    #endif
    for(i=0;i<model->BSIM3v32InstCount;i+=NSIMD)
    {
    	VecNi indexes;
	#ifdef USE_OMP
	double locTimeStep = *timeStep;
	#endif
	for(int k=0;k<NSIMD;k++)
		indexes[k] = model->BSIM3v32InstanceArray[i+k]->BSIM3v32qb;
	vecN_CKTterr(indexes,ckt,TIMESTEPptr);
	for(int k=0;k<NSIMD;k++)
		indexes[k] = model->BSIM3v32InstanceArray[i+k]->BSIM3v32qg;
	vecN_CKTterr(indexes,ckt,TIMESTEPptr);
	for(int k=0;k<NSIMD;k++)
		indexes[k] = model->BSIM3v32InstanceArray[i+k]->BSIM3v32qd;
	vecN_CKTterr(indexes,ckt,TIMESTEPptr);
	#ifdef USE_OMP
	reduTimeStep = fmin(reduTimeStep,locTimeStep);
	#endif
    }
    #ifdef USE_OMP
    *timeStep = reduTimeStep;
    i=model->BSIM3v32InstCount & (NSIMD-1);
    #endif
    
    /* less than NSIMD devices left: not worth to use openMP ? */
    for(;i<model->BSIM3v32InstCount;i++)
    {
            CKTterr(model->BSIM3v32InstanceArray[i]->BSIM3v32qb,ckt,timeStep);
            CKTterr(model->BSIM3v32InstanceArray[i]->BSIM3v32qg,ckt,timeStep);
            CKTterr(model->BSIM3v32InstanceArray[i]->BSIM3v32qd,ckt,timeStep);

    }
    
    return(OK);
}
