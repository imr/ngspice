/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "capdefs.h"
#include "sperror.h"
#include "ifsim.h"
#include "suffix.h"


/* ARGSUSED */
int
CAPmAsk(ckt,inst,which,value)
    CKTcircuit *ckt;
    GENmodel *inst;
    int which;
    IFvalue *value;
{
    CAPmodel *here = (CAPmodel*)inst;
    switch(which) {
        case CAP_MOD_CJ:
            value->rValue = here->CAPcj;
            return(OK);
        case CAP_MOD_CJSW:
            value->rValue = here->CAPcjsw;
            return(OK);
        case CAP_MOD_DEFWIDTH:
            value->rValue = here->CAPdefWidth;
            return(OK);
        case CAP_MOD_NARROW:
            value->rValue = here->CAPnarrow;
            return(OK);
        default:  
            return(E_BADPARM);
    }
}
