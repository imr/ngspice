/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "inddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
INDparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    INDinstance *here = (INDinstance*)inst;

    NG_IGNORE(select);

    switch(param) {
    case IND_IND:
        here->INDinductinst = here->INDinduct = value->rValue;
        if (!here->INDmGiven)
            here->INDm =1.0;
        here->INDindGiven = TRUE;
        break;
    case IND_TEMP:
        here->INDtemp = value->rValue + CONSTCtoK;
        here->INDtempGiven = TRUE;
        break;
    case IND_DTEMP:
        here->INDdtemp = value->rValue;
        here->INDdtempGiven = TRUE;
        break;
    case IND_M:
        here->INDm = value->rValue;
        here->INDmGiven = TRUE;
        break;
    case IND_TC1:
        here->INDtc1 = value->rValue;
        here->INDtc1Given = TRUE;
        break;
    case IND_TC2:
        here->INDtc2 = value->rValue;
        here->INDtc2Given = TRUE;
        break;
    case IND_SCALE:
        here->INDscale = value->rValue;
        here->INDscaleGiven = TRUE;
        break;
    case IND_NT:
        here->INDnt = value->rValue;
        here->INDntGiven = TRUE;
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
