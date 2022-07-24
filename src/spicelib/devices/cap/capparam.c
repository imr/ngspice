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
#include "ngspice/fteext.h"


/* ARGSUSED */
int
CAPparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    double scale;

    CAPinstance *here = (CAPinstance*)inst;

    NG_IGNORE(select);

    if (!cp_getvar("scale", CP_REAL, &scale, 0))
        scale = 1;

    switch(param) {
    case CAP_CAP:
        here->CAPcapacinst = here->CAPcapac = value->rValue;
        if (!here->CAPmGiven)
            here->CAPm = 1.0;
        here->CAPcapGiven = TRUE;
        break;
    case CAP_IC:
        here->CAPinitCond = value->rValue;
        here->CAPicGiven = TRUE;
        break;
    case CAP_TEMP:
        here->CAPtemp = value->rValue + CONSTCtoK;
        here->CAPtempGiven = TRUE;
        break;
    case CAP_DTEMP:
        here->CAPdtemp = value->rValue;
        here->CAPdtempGiven = TRUE;
        break;
    case CAP_WIDTH:
        here->CAPwidth = value->rValue * scale;
        here->CAPwidthGiven = TRUE;
        break;
    case CAP_LENGTH:
        here->CAPlength = value->rValue * scale;
        here->CAPlengthGiven = TRUE;
        break;
    case CAP_M:
        here->CAPm = value->rValue;
        here->CAPmGiven = TRUE;
        break;
    case CAP_SCALE:
        here->CAPscale = value->rValue;
        here->CAPscaleGiven = TRUE;
        break;
    case CAP_CAP_SENS:
        here->CAPsenParmNo = value->iValue;
        break;
    case CAP_TC1:
        here->CAPtc1 = value->rValue;
        here->CAPtc1Given = TRUE;
        break;
    case CAP_TC2:
        here->CAPtc2 = value->rValue;
        here->CAPtc2Given = TRUE;
        break;
    case CAP_BV_MAX:
        here->CAPbv_max = value->rValue;
        here->CAPbv_maxGiven = TRUE;
        break;
    default:
        return(E_BADPARM);
    }
    return(OK);
}
