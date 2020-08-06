/**** BSIM3v3.3.0, Released by Xuemei Xi 07/29/2005 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b3trunc.c of BSIM3v3.3.0
 * Author: 1995 Min-Chie Jeng and Mansun Chan. 
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "bsim3def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

#include "ngspice/SIMD/simdvector.h"
#include "ngspice/SIMD/simdckt.h"
#include <float.h>


int
BSIM3SIMDtrunc(
GENmodel *inModel,
CKTcircuit *ckt,
double *timeStep)
{
BSIM3model *model = (BSIM3model*)inModel;
BSIM3instance *here;
int i;
#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */
    for(i=0;i<model->BSIM3InstCount;i+=NSIMD)
	 {
	 VecNi indexes;
#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */

	 for(int k=0;k<NSIMD;k++)
		indexes[k] = model->BSIM3InstanceArray[i+k]->BSIM3qb;
	 vecN_CKTterr(indexes,ckt,timeStep);
	 for(int k=0;k<NSIMD;k++)
		indexes[k] = model->BSIM3InstanceArray[i+k]->BSIM3qg;
	 vecN_CKTterr(indexes,ckt,timeStep);
	 for(int k=0;k<NSIMD;k++)
		indexes[k] = model->BSIM3InstanceArray[i+k]->BSIM3qd;
	 vecN_CKTterr(indexes,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
	    {  printf("device %s reduces step from %g to %g\n",
                       model->BSIM3v32InstanceArray[i]->BSIM3name,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    
    for(;i<model->BSIM3InstCount;i++)
    {
            CKTterr(model->BSIM3InstanceArray[i]->BSIM3qb,ckt,timeStep);
            CKTterr(model->BSIM3InstanceArray[i]->BSIM3qg,ckt,timeStep);
            CKTterr(model->BSIM3InstanceArray[i]->BSIM3qd,ckt,timeStep);
    }
    return(OK);
}
