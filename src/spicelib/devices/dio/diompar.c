/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "diodefs.h"
#include "sperror.h"
#include "suffix.h"


int
DIOmParam(param,value,inModel)
    int param;
    IFvalue *value;
    GENmodel *inModel;
{
    DIOmodel *model = (DIOmodel*)inModel;
    switch(param) {
        case DIO_MOD_IS:
            model->DIOsatCur = value->rValue;
            model->DIOsatCurGiven = TRUE;
            break;
        case DIO_MOD_TNOM:
            model->DIOnomTemp = value->rValue+CONSTCtoK;
            model->DIOnomTempGiven = TRUE;
            break;
        case DIO_MOD_RS:
            model->DIOresist = value->rValue;
            model->DIOresistGiven = TRUE;
            break;
        case DIO_MOD_N:
            model->DIOemissionCoeff = value->rValue;
            model->DIOemissionCoeffGiven = TRUE;
            break;
        case DIO_MOD_TT:
            model->DIOtransitTime = value->rValue;
            model->DIOtransitTimeGiven = TRUE;
            break;
        case DIO_MOD_CJO:
            model->DIOjunctionCap = value->rValue;
            model->DIOjunctionCapGiven = TRUE;
            break;
        case DIO_MOD_VJ:
            model->DIOjunctionPot = value->rValue;
            model->DIOjunctionPotGiven = TRUE;
            break;
        case DIO_MOD_M:
            model->DIOgradingCoeff = value->rValue;
            model->DIOgradingCoeffGiven = TRUE;
            break;
        case DIO_MOD_EG:
            model->DIOactivationEnergy = value->rValue;
            model->DIOactivationEnergyGiven = TRUE;
            break;
        case DIO_MOD_XTI:
            model->DIOsaturationCurrentExp = value->rValue;
            model->DIOsaturationCurrentExpGiven = TRUE;
            break;
        case DIO_MOD_FC:
            model->DIOdepletionCapCoeff = value->rValue;
            model->DIOdepletionCapCoeffGiven = TRUE;
            break;
        case DIO_MOD_BV:
            model->DIObreakdownVoltage = value->rValue;
            model->DIObreakdownVoltageGiven = TRUE;
            break;
        case DIO_MOD_IBV:
            model->DIObreakdownCurrent = value->rValue;
            model->DIObreakdownCurrentGiven = TRUE;
            break;
        case DIO_MOD_D:
            /* no action - we already know we are a diode, but this */
            /* makes life easier for spice-2 like parsers */
            break;
	case DIO_MOD_KF:
	    model->DIOfNcoef = value->rValue;
	    model->DIOfNcoefGiven = TRUE;
	    break;
	case DIO_MOD_AF:
	    model->DIOfNexp = value->rValue;
	    model->DIOfNexpGiven = TRUE;
	    break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
