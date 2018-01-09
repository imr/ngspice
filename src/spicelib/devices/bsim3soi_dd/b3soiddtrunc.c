/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soiddtrunc.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMDD2.1 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "b3soidddef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
B3SOIDDtrunc(GENmodel *inModel, CKTcircuit *ckt, double *timeStep)
{
B3SOIDDmodel *model = (B3SOIDDmodel*)inModel;
B3SOIDDinstance *here;

#ifdef STEPDEBUG
    double debugtemp;
#endif /* STEPDEBUG */

    for (; model != NULL; model = B3SOIDDnextModel(model))
    {    for (here = B3SOIDDinstances(model); here != NULL;
	      here = B3SOIDDnextInstance(here))
	 {

#ifdef STEPDEBUG
            debugtemp = *timeStep;
#endif /* STEPDEBUG */
            CKTterr(here->B3SOIDDqb,ckt,timeStep);
            CKTterr(here->B3SOIDDqg,ckt,timeStep);
            CKTterr(here->B3SOIDDqd,ckt,timeStep);
#ifdef STEPDEBUG
            if(debugtemp != *timeStep)
	    {  printf("device %s reduces step from %g to %g\n",
                       here->B3SOIDDname,debugtemp,*timeStep);
            }
#endif /* STEPDEBUG */
        }
    }
    return(OK);
}



