/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include <stdio.h>
#include "const.h"
#include "cktdefs.h"
#include "ifsim.h"
#include "diodefs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
DIOmAsk (ckt,inModel,which, value)
    CKTcircuit *ckt;
    int which;
    IFvalue *value;
    GENmodel *inModel;
{
    DIOmodel *model = (DIOmodel*)inModel;
    switch (which) {
        case DIO_MOD_IS:
            value->rValue = model->DIOsatCur;
            return(OK);
        case DIO_MOD_TNOM:
            value->rValue = model->DIOnomTemp-CONSTCtoK;
            return(OK);
        case DIO_MOD_RS:
            value->rValue = model->DIOresist;
            return(OK);
        case DIO_MOD_N:
            value->rValue = model->DIOemissionCoeff;
            return(OK);
        case DIO_MOD_TT:
            value->rValue = model->DIOtransitTime;
            return(OK);
        case DIO_MOD_CJO:
            value->rValue = model->DIOjunctionCap;
            return(OK);
        case DIO_MOD_VJ:
            value->rValue = model->DIOjunctionPot;
            return(OK);
        case DIO_MOD_M:
            value->rValue = model->DIOgradingCoeff;
            return(OK);
        case DIO_MOD_EG:
            value->rValue = model->DIOactivationEnergy;
            return (OK);
        case DIO_MOD_XTI:
            value->rValue = model->DIOsaturationCurrentExp;
            return(OK);
        case DIO_MOD_FC:
            value->rValue = model->DIOdepletionCapCoeff;
            return(OK);
        case DIO_MOD_KF:
            value->rValue = model->DIOfNcoef;
            return(OK);
        case DIO_MOD_AF:
            value->rValue = model->DIOfNexp;
            return(OK);
        case DIO_MOD_BV:
            value->rValue = model->DIObreakdownVoltage;
            return(OK);
        case DIO_MOD_IBV:
            value->rValue = model->DIObreakdownCurrent;
            return(OK);
        case DIO_MOD_COND:
            value->rValue = model->DIOconductance;
            return(OK);
        default:
            return(E_BADPARM);
        }
}

