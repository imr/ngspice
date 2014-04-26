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
MUTparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    MUTinstance *here = (MUTinstance*)inst;

    NG_IGNORE(select);

    switch(param) {
        case MUT_COEFF:
            here->MUTcoupling = value->rValue;
            here->MUTindGiven = TRUE;
            break;
        case MUT_IND1:
            here->MUTindName1 = value->uValue;
            break;
        case MUT_IND2:
            here->MUTindName2 = value->uValue;
            break;
        case MUT_COEFF_SENS:
        here->MUTsenParmNo = value->iValue;
        break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
