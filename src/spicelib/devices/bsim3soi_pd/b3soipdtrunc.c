/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipdtrunc.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.2.3  02/3/5  Pin Su 
 * BSIMPD2.2.3 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soipddef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B3SOIPDtrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
B3SOIPDmodel *model = (B3SOIPDmodel*)inModel;
B3SOIPDinstance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = B3SOIPDnextModel(model))
    {    for (here = B3SOIPDinstances(model); here != NULL;
	      here = B3SOIPDnextInstance(here))
	 {
#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->B3SOIPDqb,ckt,timeStep);
            CKTterr(here->B3SOIPDqg,ckt,timeStep);
            CKTterr(here->B3SOIPDqd,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
	    {  printf("device %s reduces step from %g to %g\n",
                       here->B3SOIPDname,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}
