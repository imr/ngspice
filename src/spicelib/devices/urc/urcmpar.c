/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "urcdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
URCmParam(int param, IFvalue *value, GENmodel *inModel)
{
    URCmodel *model = (URCmodel *)inModel;
    switch(param) {
        case URC_MOD_K:
            model->URCk = value->rValue;
            model->URCkGiven = TRUE;
            break;
        case URC_MOD_FMAX:
            model->URCfmax = value->rValue;
            model->URCfmaxGiven = TRUE;
            break;
        case URC_MOD_RPERL:
            model->URCrPerL = value->rValue;
            model->URCrPerLGiven = TRUE;
            break;
        case URC_MOD_CPERL:
            model->URCcPerL = value->rValue;
            model->URCcPerLGiven = TRUE;
            break;
        case URC_MOD_ISPERL:
            model->URCisPerL = value->rValue;
            model->URCisPerLGiven = TRUE;
            break;
        case URC_MOD_RSPERL:
            model->URCrsPerL = value->rValue;
            model->URCrsPerLGiven = TRUE;
            break;
        case URC_MOD_URC:
            /* no operation - already know we are a URC, but this makes*/
            /* spice-2 like parsers happy */
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
