/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTsetBreak(ckt,time)
     *   add the given time to the breakpoint table for the given circuit
     */

#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "ifsim.h"
#include "sperror.h"



int
CKTsetBreak(CKTcircuit *ckt, double time)
{
    double *tmp;
    int i,j;

    if(ckt->CKTtime > time) {
        (*(SPfrontEnd->IFerror))(ERR_PANIC,"breakpoint in the past - HELP!",
                (IFuid *)NULL);
        return(E_INTERN);
    }
    for(i=0;i<ckt->CKTbreakSize;i++) {
        if(*(ckt->CKTbreaks+i)>time) { /* passed */
            if((*(ckt->CKTbreaks+i)-time) <= ckt->CKTminBreak) {
                /* very close together - take earlier point */
                *(ckt->CKTbreaks+i) = time;
                return(OK);
            }
            if(time-*(ckt->CKTbreaks+i-1) <= ckt->CKTminBreak) {
                /* very close together, but after, so skip */
                return(OK);
            }
            /* fits in middle - new array & insert */
            tmp = (double *)MALLOC((ckt->CKTbreakSize+1)*sizeof(double));
            if(tmp == (double *)NULL) return(E_NOMEM);
            for(j=0;j<i;j++) {
                *(tmp+j) = *(ckt->CKTbreaks+j);
            }
            *(tmp+i)=time;
            for(j=i;j<ckt->CKTbreakSize;j++) {
                *(tmp+j+1) = *(ckt->CKTbreaks+j);
            }
            FREE(ckt->CKTbreaks);
            ckt->CKTbreakSize++;
            ckt->CKTbreaks=tmp;
            return(OK);
        }
    }
    /* never found it - beyond end of time - extend out idea of time */
    if(time-ckt->CKTbreaks[ckt->CKTbreakSize-1]<=ckt->CKTminBreak) {
        /* very close tegether - keep earlier, throw out new point */
        return(OK);
    }
    /* fits at end - grow array & add on */
    ckt->CKTbreaks = (double *)REALLOC(ckt->CKTbreaks,
            (ckt->CKTbreakSize+1)*sizeof(double));
    ckt->CKTbreakSize++;
    ckt->CKTbreaks[ckt->CKTbreakSize-1]=time;
    return(OK);
}
