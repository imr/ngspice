/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 VERSION : HiSIM 2.6.1 
 FILE : hsm2trunc.c

 date : 2012.4.6

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hsm2def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

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


