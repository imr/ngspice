/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1992 David A. Gates, UC Berkeley CADgroup
**********/

    /* CKTpartition(ckt)
     * this labels each instance of a circuit as belonging to a 
     * particular processor in a multiprocessor computer.
     */

#include "ngspice.h"
#include <stdio.h>
#include "smpdefs.h"
#include "cktdefs.h"
#include "const.h"
#include "devdefs.h"
#include "sperror.h"



extern SPICEdev *DEVices[];

int
CKTpartition(register CKTcircuit *ckt)
{
    register int i, instNum = 0;
    register GENmodel *model;
    register GENinstance *inst;

    for (i=0;i<DEVmaxnum;i++) {
        if ( (ckt->CKThead[i] != NULL) ) {
	    for (model = ckt->CKThead[i]; model; model = model->GENnextModel) {
		for (inst = model->GENinstances; inst;
			inst = inst->GENnextInstance) {
		    inst->GENowner = instNum % ARCHsize;
                    instNum++;
		}
	    }
        }
    }
    return(OK);
}
