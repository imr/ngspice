/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/devdefs.h"


/* CKTinst2Node
 *  get the name and node pointer for a node given a device it is
 * bound to and the terminal of the device.
 */

int
CKTinst2Node(CKTcircuit *ckt, void *instPtr, int terminal, CKTnode **node, IFuid *nodeName)
{
    GENinstance *inst = (GENinstance *) instPtr;
    int type = inst->GENmodPtr->GENmodType;

    CKTnode *here;

    if (*(DEVices[type]->DEVpublic.terms) >= terminal && terminal > 0) {
        /* argh, terminals are counted from 1 */
        int nodenum = GENnode(inst)[terminal - 1];

        for (here = ckt->CKTnodes; here; here = here->next)
            if (here->number == nodenum) {
                *node = here;
                *nodeName = here->name;
                return OK;
            }

        return E_NOTFOUND;
    } else {
        return E_NOTERM;
    }
}
