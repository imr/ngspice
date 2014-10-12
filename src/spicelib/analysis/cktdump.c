/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles, 1991 David A. Gates
**********/

/*
TODO:
 Ngspice and cider integration note:
 This file must be changed to be consistent with the ngspice interface.
 The SPICEdev structure must be changed by adding a pointer to DEVdump 
 and DEVacct routines (as suggested below).
 Paolo Nenzi Dec 2001
 No more sure about this notice (2003)
 */

    /* CKTdump(ckt)
     * this is a simple program to dump the rhs vector to stdout
     */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"

#ifdef CIDER
/* Begin cider integration */
#include "ngspice/gendefs.h"
#include "ngspice/devdefs.h"
 
/* End cider integration */
#endif

void
CKTdump(CKTcircuit *ckt, double ref, runDesc *plot)
{
    IFvalue refData;
    IFvalue valData;
#ifdef CIDER    
    int i;
#endif

    refData.rValue = ref;
    valData.v.numValue = ckt->CKTmaxEqNum-1;
    valData.v.vec.rVec = ckt->CKTrhsOld+1;
    SPfrontEnd->OUTpData (plot, &refData, &valData);

#ifdef CIDER
/* 
 * Begin cider integration: 
 * This code has been hacked changing the SPICEdev structure.
 * SPICEdev now has DEVdump and DEVacct as members. The 
 * following code works for any devices but as of this 
 * comment is written, only numerical devices have DEVdump
 * and DEVacct routine (NUMD, NBJT, NUMD2, NBJT2, NUMOS).
 */
 
  for (i=0; i<DEVmaxnum; i++) {
    if ( DEVices[i] && DEVices[i]->DEVdump && ckt->CKThead[i] ) {
                DEVices[i]->DEVdump (ckt->CKThead[i], ckt);
    }
  }
/* End cider integration */
#endif /* CIDER */

}

#ifdef CIDER
/*
 * Routine to dump statistics about numerical devices
 * 
 * The following lines are historical:
 * This is inefficient, because we have to look up the indices
 * of the devices in the device table.  Would be simpler if
 * DEVices had an entry for an accounting function, so that indirection
 * could be used.
 * ------------------
 * The SPICEdev structure has been hacked and now has a DEVacct entry.
 */
 
void
NDEVacct(CKTcircuit *ckt, FILE *file)
{
    int i;

    if ( !ckt->CKTisSetup ) {
      return;
    }

     for (i=0; i<DEVmaxnum; i++) {
        if ( DEVices[i] && DEVices[i]->DEVacct && ckt->CKThead[i] ) {
                DEVices[i]->DEVacct (ckt->CKThead[i], ckt, file);
    }
   }
    return;
}

/* End cider integration */
#endif /* CIDER */

