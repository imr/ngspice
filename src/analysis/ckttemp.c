/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /* CKTtemp(ckt)
     * this is a driver program to iterate through all the various
     * temperature dependency functions provided for the circuit 
     * elements in the given circuit 
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
CKTtemp(register CKTcircuit *ckt)
{
    int error;
    register int i;

    ckt->CKTvt = CONSTKoverQ * ckt->CKTtemp;

    for (i=0;i<DEVmaxnum;i++) {
        if ( ((*DEVices[i]).DEVtemperature != NULL) &&
                (ckt->CKThead[i] != NULL) ){
            error = (*((*DEVices[i]).DEVtemperature))(ckt->CKThead[i],ckt);
            if(error) return(error);
        }
    }
    return(OK);
}
