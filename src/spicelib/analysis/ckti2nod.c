/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

    /* CKTinst2Node
     *  get the name and node pointer for a node given a device it is
     * bound to and the terminal of the device.
     */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/devdefs.h"



int
CKTinst2Node(CKTcircuit *ckt, void *instPtr, int terminal, CKTnode **node, IFuid *nodeName)
{
    int nodenum;
    int type;
    CKTnode *here;

    type = ((GENinstance *)instPtr)->GENmodPtr->GENmodType;

    if(*(DEVices[type]->DEVpublic.terms) >= terminal && terminal > 0) {
        /* argh, terminals are counted from 1 */
        nodenum = ((GENinstance *)instPtr)->GENnode[terminal - 1];
        /* ok, now we know its number, so we just have to find it.*/
        for(here = ckt->CKTnodes;here;here = here->next) {
            if(here->number == nodenum) {
                /* found it */
                *node = here;
                *nodeName = here->name;
                return(OK);
            }
        }
        return(E_NOTFOUND);
    } else {
        return(E_NOTERM);
    }
}
