/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /*
     *CKTsetNodPm
     *
     *   set a parameter on a node.
     */

#include "ngspice.h"
#include "ifsim.h"
#include "iferrmsg.h"
#include "cktdefs.h"



/* ARGSUSED */
int
CKTsetNodPm(CKTcircuit *ckt, CKTnode *node, int parm, IFvalue *value, IFvalue *selector)
{
    IGNORE(ckt);
    IGNORE(selector);

    if(!node) return(E_BADPARM);
    switch(parm) {

    case PARM_NS:
        node->nodeset = value->rValue;
        node->nsGiven = 1;
        break;

    case PARM_IC:
        node->ic = value->rValue;
        node->icGiven = 1;
        break;

    case PARM_NODETYPE:
        node->type = value->iValue;
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}
