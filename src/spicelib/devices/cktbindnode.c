/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* CKTbindNode
 *
 * bind a node of the specified device of the given type to its place
 * in the specified circuit.  */

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/devdefs.h"
#include "ngspice/sperror.h"

#include "dev.h"


int
CKTbindNode(CKTcircuit *ckt, GENinstance *instance, int term, CKTnode *node)
{
    SPICEdev **devs = devices();
    int type = instance->GENmodPtr->GENmodType;

    NG_IGNORE(ckt);

    if (*(devs[type]->DEVpublic.terms) >= term && term > 0) {
        /* argh, terminals are counted from 1 */
        GENnode(instance)[term - 1] = node->number;
        return OK;
    } else {
        return E_NOTERM;
    }
}
