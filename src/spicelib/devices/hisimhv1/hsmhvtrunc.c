/***********************************************************************

 HiSIM (Hiroshima University STARC IGFET Model)
 Copyright (C) 2012 Hiroshima University & STARC

 MODEL NAME : HiSIM_HV 
 ( VERSION : 1  SUBVERSION : 2  REVISION : 4 )
 Model Parameter VERSION : 1.23
 FILE : hsmhvtrunc.c

 DATE : 2013.04.30

 released by 
                Hiroshima University &
                Semiconductor Technology Academic Research Center (STARC)
***********************************************************************/

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hsmhvdef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int HSMHVtrunc(
     GENmodel *inModel,
     register CKTcircuit *ckt,
     double *timeStep)

{
  register HSMHVmodel *model = (HSMHVmodel*)inModel;
  register HSMHVinstance *here;
#ifdef STEPDEBUG
  double debugtemp=0.0 ;
#endif /* STEPDEBUG */
  
  for ( ;model != NULL ;model = HSMHVnextModel(model)) {
    for ( here=HSMHVinstances(model);here!=NULL ;
	  here = HSMHVnextInstance(here)) {
#ifdef STEPDEBUG
      debugtemp = *timeStep;
#endif /* STEPDEBUG */
      CKTterr(here->HSMHVqb,ckt,timeStep);
      CKTterr(here->HSMHVqg,ckt,timeStep);
      CKTterr(here->HSMHVqd,ckt,timeStep);

      CKTterr(here->HSMHVqbs,ckt,timeStep);
      CKTterr(here->HSMHVqbd,ckt,timeStep);
      CKTterr(here->HSMHVqfd,ckt,timeStep);
      CKTterr(here->HSMHVqfs,ckt,timeStep);


#ifdef STEPDEBUG
      if ( debugtemp != *timeStep ) 
	printf("device %s reduces step from %g to %g\n",
	       here->HSMHVname, debugtemp, *timeStep);
#endif /* STEPDEBUG */
    }
  }
  return(OK);
}
