/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "ifsim.h"
#include "inddefs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
INDparam(param,value,inst,select)
    int param;
    IFvalue *value;
    GENinstance *inst;
    IFvalue *select;
{
    INDinstance *here = (INDinstance*)inst;
    switch(param) {
        case IND_IND:
            here->INDinduct = value->rValue;
            here->INDindGiven = TRUE;
            break;
        case IND_IC:
            here->INDinitCond = value->rValue;
            here->INDicGiven = TRUE;
            break;
        case IND_IND_SENS:
            here->INDsenParmNo = value->iValue;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
