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
CKTsetNodPm(void *ckt, void *node, int parm, IFvalue *value, IFvalue *selector)
{
    if(!node) return(E_BADPARM);
    switch(parm) {

    case PARM_NS:
        ((CKTnode *)node)->nodeset = value->rValue;
        ((CKTnode *)node)->nsGiven = 1;
        break;

    case PARM_IC:
        ((CKTnode *)node)->ic = value->rValue;
        ((CKTnode *)node)->icGiven = 1;
        break;

    case PARM_NODETYPE:
        ((CKTnode *)node)->type = value->iValue;
        break;

    default:
        return(E_BADPARM);
    }
    return(OK);
}
