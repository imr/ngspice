/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "resdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/missing_math.h"
#include "ngspice/fteext.h"

int
RESparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    double scale;

    RESinstance *here = (RESinstance *)inst;

    NG_IGNORE(select);

    if (!cp_getvar("scale", CP_REAL, &scale, 0))
        scale = 1;

    switch(param) {
    case RES_TEMP:
        here->REStemp = value->rValue + CONSTCtoK;
        if (here->REStemp < 1e-6)
            here->REStemp = 0;
        here->REStempGiven = TRUE;
        break;
    case RES_DTEMP:
        here->RESdtemp = value->rValue;
        here->RESdtempGiven = TRUE;
        break;
    case RES_RESIST:
        /* 0 valued resistor causes ngspice to hang -- can't solve for initial voltage */
        if ( AlmostEqualUlps( value->rValue, 0, 3 ) ) value->rValue = 0.001; /* 0.001 should be sufficiently small */
        here->RESresist = value->rValue;
        here->RESresGiven = TRUE;
        break;
    case RES_ACRESIST:
        here->RESacResist = value->rValue;
        here->RESacresGiven = TRUE;
        break;
    case RES_WIDTH:
        here->RESwidth = value->rValue * scale;
        here->RESwidthGiven = TRUE;
        break;
    case RES_LENGTH:
        here->RESlength = value->rValue * scale;
        here->RESlengthGiven = TRUE;
        break;
    case RES_SCALE:
        here->RESscale = value->rValue;
        here->RESscaleGiven = TRUE;
        break;
    case RES_RESIST_SENS:
        here->RESsenParmNo = value->iValue;
        break;
    case RES_M:
        here->RESm = value->rValue;
        here->RESmGiven = TRUE;
        break;
    case RES_TC1:
        here->REStc1 = value->rValue;
        here->REStc1Given = TRUE;
        break;
    case RES_TC2:
        here->REStc2 = value->rValue;
        here->REStc2Given = TRUE;
        break;
    case RES_TCE:
        here->REStce = value->rValue;
        here->REStceGiven = TRUE;
        break;
    case RES_NOISY:
        here->RESnoisy = value->iValue;
        here->RESnoisyGiven = TRUE;
        break;
    case RES_BV_MAX:
        here->RESbv_max = value->rValue;
        here->RESbv_maxGiven = TRUE;
        break;
    default:
        return(E_BADPARM);
    }
    RESupdate_conduct(here, FALSE);
    return(OK);
}
