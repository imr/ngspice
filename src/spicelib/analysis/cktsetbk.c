/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTsetBreak(ckt,time)
     *   add the given time to the breakpoint table for the given circuit
     */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"

/* define to enable breakpoint trace code */
/* #define TRACE_BREAKPOINT */

int
CKTsetBreak(CKTcircuit *ckt, double time)
{
    double *tmp;
    int i,j;

#ifdef TRACE_BREAKPOINT
    printf("[t:%e] \t want breakpoint for t = %e\n", ckt->CKTtime, time);
#endif

    /* If time equals ckt->CKTtime, but differences due to
       limtations of double precision exist */
    if (AlmostEqualUlps(time, ckt->CKTtime, 3)) {
#ifdef TRACE_BREAKPOINT // #if (1)
        fprintf(stderr, "Warning: Setting a new breakpoint at %e is ignored,\n    as current time is %e\n", time, ckt->CKTtime);
#endif
        return (OK);
    }

    if(ckt->CKTtime > time) {
        SPfrontEnd->IFerrorf (ERR_PANIC, "breakpoint in the past - HELP!");
        return(E_INTERN);
    }
    for(i=0;i<ckt->CKTbreakSize;i++) {
        if(ckt->CKTbreaks[i]>time) { /* passed */
            if((ckt->CKTbreaks[i]-time) <= ckt->CKTminBreak) {
                /* very close together - take earlier point */
#ifdef TRACE_BREAKPOINT
                printf("[t:%e] \t %e replaces %e\n", ckt->CKTtime, time,
		ckt->CKTbreaks[i]);
		CKTbreakDump(ckt);
#endif		
                ckt->CKTbreaks[i] = time;
                return(OK);
            }
            if(i>0 && time-ckt->CKTbreaks[i-1] <= ckt->CKTminBreak) {
                /* very close together, but after, so skip */
#ifdef TRACE_BREAKPOINT
                printf("[t:%e] \t %e skipped\n", ckt->CKTtime, time);
		CKTbreakDump(ckt);
#endif			
                return(OK);
            }
            /* fits in middle - new array & insert */
            tmp = TMALLOC(double, ckt->CKTbreakSize + 1);
            if(tmp == NULL) return(E_NOMEM);
            for(j=0;j<i;j++) {
                tmp[j] = ckt->CKTbreaks[j];
            }
            tmp[i]=time;
#ifdef TRACE_BREAKPOINT
                printf("[t:%e] \t %e added\n", ckt->CKTtime, time);
		CKTbreakDump(ckt);
#endif	    
            for(j=i;j<ckt->CKTbreakSize;j++) {
                tmp[j+1] = ckt->CKTbreaks[j];
            }
            FREE(ckt->CKTbreaks);
            ckt->CKTbreakSize++;
            ckt->CKTbreaks=tmp;

            return(OK);
        }
    }
    /* never found it - beyond end of time - extend out idea of time */
    if(ckt->CKTbreaks && time-ckt->CKTbreaks[ckt->CKTbreakSize-1]<=ckt->CKTminBreak) {
        /* very close tegether - keep earlier, throw out new point */
#ifdef TRACE_BREAKPOINT
                printf("[t:%e] \t %e skipped (at the end)\n", ckt->CKTtime, time);
                CKTbreakDump(ckt);
#endif	
        return(OK);
    }
    /* fits at end - grow array & add on */
    ckt->CKTbreaks = TREALLOC(double, ckt->CKTbreaks, ckt->CKTbreakSize + 1);
    ckt->CKTbreakSize++;
    ckt->CKTbreaks[ckt->CKTbreakSize-1]=time;
#ifdef TRACE_BREAKPOINT
                printf("[t:%e] \t %e added at end\n", ckt->CKTtime, time);
		CKTbreakDump(ckt);
#endif    
    return(OK);
}
