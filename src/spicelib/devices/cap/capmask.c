/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "capdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/ifsim.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
CAPmAsk(CKTcircuit *ckt, GENmodel *inst, int which, IFvalue *value)
{
    CAPmodel *here = (CAPmodel*)inst;

    NG_IGNORE(ckt);

    switch(which) {
    case CAP_MOD_TNOM:
        value->rValue = here->CAPtnom-CONSTCtoK;
        return(OK);
    case CAP_MOD_TC1:
        value->rValue = here->CAPtempCoeff1;
        return(OK);
    case CAP_MOD_TC2:
        value->rValue = here->CAPtempCoeff2;
        return(OK);
    case CAP_MOD_CAP:
        value->rValue = here->CAPmCap;
        return(OK);
    case CAP_MOD_CJ:
        value->rValue = here->CAPcj;
        return(OK);
    case CAP_MOD_CJSW:
        value->rValue = here->CAPcjsw;
        return(OK);
    case CAP_MOD_DEFWIDTH:
        value->rValue = here->CAPdefWidth;
        return(OK);
    case CAP_MOD_DEFLENGTH:
        value->rValue = here->CAPdefLength;
        return(OK);
    case CAP_MOD_NARROW:
        value->rValue = here->CAPnarrow;
        return(OK);
    case CAP_MOD_SHORT:
        value->rValue = here->CAPshort;
        return(OK);
    case CAP_MOD_DEL:
        value->rValue = here->CAPdel;
        return(OK);
    case CAP_MOD_DI:
        value->rValue = here->CAPdi;
        return(OK);
    case CAP_MOD_THICK:
        value->rValue = here->CAPthick;
        return(OK);
    case CAP_MOD_BV_MAX:
        value->rValue = here->CAPbv_max;
        return(OK);
    default:
        return(E_BADPARM);
    }
}
