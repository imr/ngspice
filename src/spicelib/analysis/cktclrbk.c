/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTclrBreak(ckt)
     *   delete the first time from the breakpoint table for the given circuit
     */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"



int
CKTclrBreak(CKTcircuit *ckt)
{
    double *tmp;
    int j;

    if(ckt->CKTbreakSize >2) {
        tmp = TMALLOC(double, ckt->CKTbreakSize - 1);
        if(tmp == NULL) return(E_NOMEM);
        for(j=1;j<ckt->CKTbreakSize;j++) {
            tmp[j-1] = ckt->CKTbreaks[j];
        }
        FREE(ckt->CKTbreaks);
        ckt->CKTbreakSize--;
        ckt->CKTbreaks=tmp;
    } else {
        ckt->CKTbreaks[0] = ckt->CKTbreaks[1];
        ckt->CKTbreaks[1] = ckt->CKTfinalTime;
    }
    return(OK);
}
