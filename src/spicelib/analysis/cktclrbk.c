/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTclrBreak(ckt)
     *   delete the first time from the breakpoint table for the given circuit
     */

#include "ngspice.h"
#include "cktdefs.h"
#include "sperror.h"



int
CKTclrBreak(CKTcircuit *ckt)
{
    double *tmp;
    int j;

    if(ckt->CKTbreakSize >2) {
        tmp = (double *)MALLOC((ckt->CKTbreakSize-1)*sizeof(double));
        if(tmp == (double *)NULL) return(E_NOMEM);
        for(j=1;j<ckt->CKTbreakSize;j++) {
            *(tmp+j-1) = *(ckt->CKTbreaks+j);
        }
        FREE(ckt->CKTbreaks);
        ckt->CKTbreakSize--;
        ckt->CKTbreaks=tmp;
    } else {
        *(ckt->CKTbreaks)= *(ckt->CKTbreaks+1);
        *(ckt->CKTbreaks+1) = ckt->CKTfinalTime;
    }
    return(OK);
}
