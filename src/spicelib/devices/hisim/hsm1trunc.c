/***********************************************************************
 HiSIM v1.1.0
 File: hsm1trunc.c of HiSIM v1.1.0

 Copyright (C) 2002 STARC

 June 30, 2002: developed by Hiroshima University and STARC
 June 30, 2002: posted by Keiichi MORIKAWA, STARC Physical Design Group
***********************************************************************/

/*
 * Modified by Paolo Nenzi 2002
 * ngspice integration
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "hsm1def.h"
#include "sperror.h"
#include "suffix.h"

int HSM1trunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
HSM1model *model = (HSM1model*)inModel;
HSM1instance *here;
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


