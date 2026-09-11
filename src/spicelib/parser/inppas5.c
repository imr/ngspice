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

#include "inppas5.h"
#include "inpxx.h"

#ifdef XSPICE
/* gtri - add - wbk - 11/9/90 - include function prototypes */
#include "ngspice/mifproto.h"
/* gtri - end - wbk - 11/9/90 */
#endif


/* uncomment to trace in this file */
/*#define TRACE*/

/* pass 5 - If option rshunt is given,
add a resistor to each voltage node */
void INPpas5(CKTcircuit *ckt, INPtables *tab)
{
    CKTnode* node;
    int mytype = -1;
    IFuid uid;           /* uid for default res model */
    int error;           /* error code temporary */
    GENinstance* fast;   /* pointer to the actual instance */
    IFvalue ptemp;       /* a value structure to package resistance into */
    int nadded = 0;      /* resistors added */
    double rsval = 0.;   /* rshunt resistance value */

    /* get the cshunt value */
    if (!cp_getvar("rshunt_value", CP_REAL, &rsval, 0))
        return;

    if ((mytype = INPtypelook("Resistor")) < 0) {
        fprintf(stderr, "Device type Resistor not supported by this binary\n");
        return;
    }

    if (!tab->defRmod) {    /* create default C model */
        IFnewUid(ckt, &uid, NULL, "R", UID_MODEL, NULL);
        error = (*(ft_sim->newModel))(ckt, mytype, &(tab->defRmod), uid);
    }

    /* scan through all nodes, add a new R device for each voltage node */
    for (node = ckt->CKTnodes; node; node = node->next) {
        if (node->type == NODE_VOLTAGE && (node->number > 0)) {
            int nn = node->number;
            char* devname = tprintf("resist%dshunt", nn);

            fast = (*(ft_sim->findInstance))(ckt, devname);
            if (fast) {
                fprintf(stderr,
                    "WARNING: non-rshunt instance %s already exists\n",
                    devname);
                tfree(devname);
                continue;
            }
            error = (*(ft_sim->newInstance))(ckt, tab->defRmod, &fast, devname);
            if (error) {
                fprintf(stderr,
                    "ERROR: rshunt newInstance status %d devname %s\n",
                    error, devname);
                tfree(devname);
                continue;
            }
            /* devname is eventually freed when INPtabEnd is called */
            INPinsert(&devname, tab);

           /* the top node, second node is gnd automatically */
            (*(ft_sim->bindNode))(ckt, fast, 1, node);

            /* value of the resistance */
            ptemp.rValue = rsval;
            error = INPpName("resistance", &ptemp, ckt, mytype, fast);

            nadded++;
        }
    }
    if (nadded == 1)
        printf("Option rshunt: 1 resistor added with %g Ohms\n", rsval);
    else
        printf("Option rshunt: %d resistors added with %g Ohms each\n", nadded, rsval);
}
