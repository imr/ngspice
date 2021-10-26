/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/inpdefs.h"
#include "ngspice/inpmacs.h"
#include "ngspice/fteext.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/cktdefs.h"

#include "inppas4.h"
#include "inpxx.h"

#ifdef XSPICE
/* gtri - add - wbk - 11/9/90 - include function prototypes */
#include "ngspice/mifproto.h"
/* gtri - end - wbk - 11/9/90 */
#endif


/* uncomment to trace in this file */
/*#define TRACE*/

/* pass 4 - If option cshunt is given,
add a capacitor to each voltage node */
void INPpas4(CKTcircuit *ckt, INPtables *tab)
{
    CKTnode* node;
    int mytype = -1;
    IFuid uid;           /* uid for default cap model */
    int error;           /* error code temporary */
    GENinstance* fast;   /* pointer to the actual instance */
    IFvalue ptemp;       /* a value structure to package capacitance into */
    int nadded = 0;      /* capacitors added */
    double csval = 0.;        /* cshunt capacitors value */

    /* get the cshunt value */
    if (!cp_getvar("cshunt_value", CP_REAL, &csval, 0))
        return;

    if ((mytype = INPtypelook("Capacitor")) < 0) {
        fprintf(stderr, "Device type Capacitor not supported by this binary\n");
        return;
    }

    if (!tab->defCmod) {    /* create default C model */
        IFnewUid(ckt, &uid, NULL, "C", UID_MODEL, NULL);
        error = (*(ft_sim->newModel))(ckt, mytype, &(tab->defCmod), uid);
    }

    /* scan through all nodes, add a new C device for each voltage node */
    for (node = ckt->CKTnodes; node; node = node->next) {
        if (node->type == NODE_VOLTAGE && (node->number > 0)) {
            int nn = node->number;
            char* devname = tprintf("capac%dshunt", nn);

            (*(ft_sim->newInstance))(ckt, tab->defCmod, &fast, devname);

           /* the top node, second node is gnd automatically */
            (*(ft_sim->bindNode))(ckt, fast, 1, node);

            /* value of the capacitance */
            ptemp.rValue = csval;
            error = INPpName("capacitance", &ptemp, ckt, mytype, fast);

            /* add device numbers for statistics */
            ckt->CKTstat->STATdevNum[mytype].instNum++;
            ckt->CKTstat->STATtotalDev++;

            nadded++;
        }
    }
    printf("Option cshunt: %d capacitors added with %g F each\n", nadded, csval);
}
