/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1992 David A. Gates, UC Berkeley CADgroup
**********/

    /* CKTpartition(ckt)
     * this labels each instance of a circuit as belonging to a 
     * particular processor in a multiprocessor computer.
     */

#include "ngspice.h"
#include "smpdefs.h"
#include "cktdefs.h"
#include "const.h"
#include "devdefs.h"
#include "sperror.h"



extern SPICEdev **DEVices;
#ifdef XSPICE
extern int *DEVicesfl;
#endif

int
CKTpartition(CKTcircuit *ckt)
{
    int i, instNum = 0;
    GENmodel *model;
    GENinstance *inst;

    for (i=0;i<DEVmaxnum;i++) {
        if ( (ckt->CKThead[i] != NULL) 
#ifdef XSPICE
&& DEVicesfl[i] == 0
#endif	
	 ){
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
