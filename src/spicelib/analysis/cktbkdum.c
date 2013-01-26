/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /*  CKTbreakDump(ckt) - dump the breakpoint table associated with
     *  the given circuit 
     */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"



void
CKTbreakDump(CKTcircuit *ckt)
{
    int i;
    for(i=0;i<ckt->CKTbreakSize;i++) {
        (void)printf("breakpoint table entry %d is %g\n",i,ckt->CKTbreaks[i]);
    }
}
