/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipdtrunc.c          98/5/01
**********/


#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "cktdefs.h"
#include "b3soipddef.h"
#include "sperror.h"
#include "suffix.h"


int
B3SOIPDtrunc (inModel, ckt, timeStep)
     GENmodel *inModel;
     register CKTcircuit *ckt;
     double *timeStep;
{
  register B3SOIPDmodel *model = (B3SOIPDmodel *) inModel;
  register B3SOIPDinstance *here;

#ifdef STEPDEBUG
  double debugtemp;
#endif /* STEPDEBUG */

  for (; model != NULL; model = model->B3SOIPDnextModel)
    {
      for (here = model->B3SOIPDinstances; here != NULL;
	   here = here->B3SOIPDnextInstance)
	{
#ifdef STEPDEBUG
	  debugtemp = *timeStep;
#endif /* STEPDEBUG */
	  CKTterr (here->B3SOIPDqb, ckt, timeStep);
	  CKTterr (here->B3SOIPDqg, ckt, timeStep);
	  CKTterr (here->B3SOIPDqd, ckt, timeStep);
#ifdef STEPDEBUG
	  if (debugtemp != *timeStep)
	    {
	      printf ("device %s reduces step from %g to %g\n",
		      here->B3SOIPDname, debugtemp, *timeStep);
	    }
#endif /* STEPDEBUG */
	}
    }
  return (OK);
}
