/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/
/*
 */

/*
 * This routine gives access to the internal device parameter
 * of TRAnsmission lines
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "ifsim.h"
#include "tradefs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
TRAask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, IFvalue *select)
{
    TRAinstance *here = (TRAinstance *)inst;
    int temp;
    double *v, *w;

    switch(which) {
        case TRA_POS_NODE1:
            value->iValue = here->TRAposNode1;
            return (OK);
        case TRA_NEG_NODE1:
            value->iValue = here->TRAnegNode1;
            return (OK);
        case TRA_POS_NODE2:
            value->iValue = here->TRAposNode2;
            return (OK);
        case TRA_NEG_NODE2:
            value->iValue = here->TRAnegNode2;
            return (OK);
        case TRA_INT_NODE1:
            value->iValue = here->TRAintNode1;
            return (OK);
        case TRA_INT_NODE2:
            value->iValue = here->TRAintNode2;
            return (OK);
        case TRA_Z0:
            value->rValue = here->TRAimped;
            return (OK);
        case TRA_TD:
            value->rValue = here->TRAtd;
            return (OK);
        case TRA_NL:
            value->rValue = here->TRAnl;
            return (OK);
        case TRA_FREQ:
            value->rValue = here->TRAf;
            return (OK);
        case TRA_V1:
            value->rValue = here->TRAinitVolt1;
            return (OK);
        case TRA_I1:
            value->rValue = here->TRAinitCur1;
            return (OK);
        case TRA_V2:
            value->rValue = here->TRAinitVolt2;
            return (OK);
        case TRA_I2:
            value->rValue = here->TRAinitCur2;
            return (OK);
        case TRA_RELTOL:
            value->rValue = here->TRAreltol;
            return (OK);
        case TRA_ABSTOL:
            value->rValue = here->TRAabstol;
            return (OK);
        case TRA_BR_EQ1:
            value->rValue = here->TRAbrEq1;
            return (OK);
        case TRA_BR_EQ2:
            value->rValue = here->TRAbrEq2;
            return (OK);
        case TRA_DELAY:
            value->v.vec.rVec = (double *) MALLOC(here->TRAsizeDelay);
            value->v.numValue = temp = here->TRAsizeDelay;
	    v = value->v.vec.rVec;
	    w = here->TRAdelays;
            while (temp--)
                *v++ = *w++;
            return (OK);
        default:
            return (E_BADPARM);
    }
    /* NOTREACHED */
}
