/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2011 Hiroshima University & STARC

 VERSION : HiSIM_2.5.1 
 FILE : hsm2trunc.c

 date : 2011.04.07

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "hsm2def.h"
#include "sperror.h"
#include "suffix.h"

int HSM2trunc(
     GENmodel *inModel,
     register CKTcircuit *ckt,
     double *timeStep)

{
  register HSM2model *model = (HSM2model*)inModel;
  register HSM2instance *here;
#ifdef STEPDEBUG
  double debugtemp;
#endif /* STEPDEBUG */
  
  for ( ;model != NULL ;model = model->HSM2nextModel ) {
    for ( here=model->HSM2instances ;here!=NULL ;
	  here = here->HSM2nextInstance ) {
#ifdef STEPDEBUG
      debugtemp = *timeStep;
#endif /* STEPDEBUG */
      CKTterr(here->HSM2qb,ckt,timeStep);
      CKTterr(here->HSM2qg,ckt,timeStep);
      CKTterr(here->HSM2qd,ckt,timeStep);
#ifdef STEPDEBUG
      if ( debugtemp != *timeStep ) 
	printf("device %s reduces step from %g to %g\n",
	       here->HSM2name, debugtemp, *timeStep);
#endif /* STEPDEBUG */
    }
  }
  return(OK);
}


