/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

/*
 * This routine gives access to the internal model parameter
 * of voltage controlled SWitch
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "swdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
SWmAsk(CKTcircuit *ckt, GENmodel *inModel, int which, IFvalue *value)
{
    SWmodel *model = (SWmodel *) inModel;

    NG_IGNORE(ckt);

    switch (which) {
    case SW_MOD_RON:
        value->rValue = model->SWonResistance;
        return OK;
    case SW_MOD_ROFF:
        value->rValue = model->SWoffResistance;
        return OK;
    case SW_MOD_VTH:
        value->rValue = model->SWvThreshold;
        return OK;
    case SW_MOD_VHYS:
        value->rValue = model->SWvHysteresis;
        return OK;
    case SW_MOD_GON:
        value->rValue = model->SWonConduct;
        return OK;
    case SW_MOD_GOFF:
        value->rValue = model->SWoffConduct;
        return OK;
    default:
        return E_BADPARM;
    }
}
