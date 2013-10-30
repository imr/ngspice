/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: September 2003 Paolo Nenzi
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "capdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
CAPmParam(int param, IFvalue *value, GENmodel *inModel)
{
    CAPmodel *mod = (CAPmodel*)inModel;
    switch(param) {
    case CAP_MOD_TNOM:
        mod->CAPtnom = value->rValue+CONSTCtoK;
        mod->CAPtnomGiven = TRUE;
        break;
    case CAP_MOD_TC1:
        mod->CAPtempCoeff1 = value->rValue;
        mod->CAPtc1Given = TRUE;
        break;
    case CAP_MOD_TC2:
        mod->CAPtempCoeff2 = value->rValue;
        mod->CAPtc2Given = TRUE;
        break;
    case CAP_MOD_CAP:
        mod->CAPmCap = value->rValue;
        mod->CAPmCapGiven = TRUE;
        break;
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
    case CAP_MOD_DEFLENGTH:
        mod->CAPdefLength = value->rValue;
        mod->CAPdefLengthGiven = TRUE;
        break;
    case CAP_MOD_NARROW:
        mod->CAPnarrow = value->rValue;
        mod->CAPnarrowGiven = TRUE;
        break;
    case CAP_MOD_SHORT:
        mod->CAPshort = value->rValue;
        mod->CAPshortGiven = TRUE;
        break;
    case CAP_MOD_DEL:
        mod->CAPdel = value->rValue;
        mod->CAPdelGiven = TRUE;
        break;
    case CAP_MOD_DI:
        mod->CAPdi = value->rValue;
        mod->CAPdiGiven = TRUE;
        break;
    case CAP_MOD_THICK:
        mod->CAPthick = value->rValue;
        mod->CAPthickGiven = TRUE;
        break;
    case CAP_MOD_BV_MAX:
        mod->CAPbv_max = value->rValue;
        mod->CAPbv_maxGiven = TRUE;
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

