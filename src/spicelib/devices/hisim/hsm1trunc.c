/***********************************************************************
 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2003 STARC

 VERSION : HiSIM 1.2.0
 FILE : hsm1trunc of HiSIM 1.2.0

 April 9, 2003 : released by STARC Physical Design Group
***********************************************************************/

#include "ngspice.h"
#include "cktdefs.h"
#include "hsm1def.h"
#include "sperror.h"
#include "suffix.h"

int 
HSM1trunc(GENmodel *inModel, register CKTcircuit *ckt, double *timeStep)
{
  register HSM1model *model = (HSM1model*)inModel;
  register HSM1instance *here;
#ifdef STEPDEBUG
  double debugtemp;
#endif /* STEPDEBUG */
  
  for ( ;model != NULL ;model = model->HSM1nextModel ) {
    for ( here=model->HSM1instances ;here!=NULL ;
	  here = here->HSM1nextInstance ) {
    	  
    if (here->HSM1owner != ARCHme)
            continue;	  
	  
#ifdef STEPDEBUG
      debugtemp = *timeStep;
#endif /* STEPDEBUG */
      CKTterr(here->HSM1qb,ckt,timeStep);
      CKTterr(here->HSM1qg,ckt,timeStep);
      CKTterr(here->HSM1qd,ckt,timeStep);
#ifdef STEPDEBUG
      if ( debugtemp != *timeStep ) 
	printf("device %s reduces step from %g to %g\n",
	       here->HSM1name, debugtemp, *timeStep);
#endif /* STEPDEBUG */
    }
  }
  return(OK);
}


