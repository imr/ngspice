/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include <stdio.h>
#include "ifsim.h"
#include "capdefs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
CAPparam(param,value,inst,select)
    int param;
    IFvalue *value;
    GENinstance *inst;
    IFvalue *select;
{
    CAPinstance *here = (CAPinstance*)inst;
    switch(param) {
        case CAP_CAP:
            here->CAPcapac = value->rValue;
            here->CAPcapGiven = TRUE;
            break;
        case CAP_IC:
            here->CAPinitCond = value->rValue;
            here->CAPicGiven = TRUE;
            break;
        case CAP_WIDTH:
            here->CAPwidth = value->rValue;
            here->CAPwidthGiven = TRUE;
            break;
        case CAP_LENGTH:
            here->CAPlength = value->rValue;
            here->CAPlengthGiven = TRUE;
            break;
        case CAP_CAP_SENS:
            here->CAPsenParmNo = value->iValue;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
