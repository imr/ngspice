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


int
CAPmParam(param,value,inModel)
    int param;
    IFvalue *value;
    GENmodel *inModel;
{
    CAPmodel *mod = (CAPmodel*)inModel;
    switch(param) {
        case CAP_MOD_CJ :
            mod->CAPcj = value->rValue;
            mod->CAPcjGiven = TRUE;
            break;
        case CAP_MOD_CJSW :
            mod->CAPcjsw = value->rValue;
            mod->CAPcjswGiven = TRUE;
            break;
        case CAP_MOD_DEFWIDTH:
            mod->CAPdefWidth = value->rValue;
            mod->CAPdefWidthGiven = TRUE;
            break;
        case CAP_MOD_NARROW:
            mod->CAPnarrow = value->rValue;
            mod->CAPnarrowGiven = TRUE;
            break;
        case CAP_MOD_C:
            /* just being reassured by the user that we are a capacitor */
            /* no-op */
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}

