/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified by Dietmar Warning 2003 and Paolo Nenzi 2003
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
DIOmParam(int param, IFvalue *value, GENmodel *inModel)
{
    DIOmodel *model = (DIOmodel*)inModel;
    switch(param) {
        case DIO_MOD_IS:
            model->DIOsatCur = value->rValue;
            model->DIOsatCurGiven = TRUE;
            break;
        case DIO_MOD_JSW:
            model->DIOsatSWCur = value->rValue;
            model->DIOsatSWCurGiven = TRUE;
            break;

        case DIO_MOD_TNOM:
            model->DIOnomTemp = value->rValue+CONSTCtoK;
            model->DIOnomTempGiven = TRUE;
            break;
        case DIO_MOD_RS:
            model->DIOresist = value->rValue;
            model->DIOresistGiven = TRUE;
            break;
        case DIO_MOD_TRS:
            model->DIOresistTemp1 = value->rValue;
            model->DIOresistTemp1Given = TRUE;
            break;
        case DIO_MOD_TRS2:
            model->DIOresistTemp2 = value->rValue;
            model->DIOresistTemp2Given = TRUE;
            break;	    	    	    
        case DIO_MOD_N:
            model->DIOemissionCoeff = value->rValue;
            model->DIOemissionCoeffGiven = TRUE;
            break;
        case DIO_MOD_TT:
            model->DIOtransitTime = value->rValue;
            model->DIOtransitTimeGiven = TRUE;
            break;
        case DIO_MOD_TTT1:
            model->DIOtranTimeTemp1 = value->rValue;
            model->DIOtranTimeTemp1Given = TRUE;
            break;	
        case DIO_MOD_TTT2:
            model->DIOtranTimeTemp2 = value->rValue;
            model->DIOtranTimeTemp2Given = TRUE;
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
        case DIO_MOD_TM1:
            model->DIOgradCoeffTemp1 = value->rValue;
            model->DIOgradCoeffTemp1Given = TRUE;
            break;
        case DIO_MOD_TM2:
            model->DIOgradCoeffTemp2 = value->rValue;
            model->DIOgradCoeffTemp2Given = TRUE;
            break;	    	    
        case DIO_MOD_CJSW:
            model->DIOjunctionSWCap = value->rValue;
            model->DIOjunctionSWCapGiven = TRUE;
            break;
        case DIO_MOD_VJSW:
            model->DIOjunctionSWPot = value->rValue;
            model->DIOjunctionSWPotGiven = TRUE;
            break;
        case DIO_MOD_MJSW:
            model->DIOgradingSWCoeff = value->rValue;
            model->DIOgradingSWCoeffGiven = TRUE;
            break;
        case DIO_MOD_IKF:
            model->DIOforwardKneeCurrent = value->rValue;
            model->DIOforwardKneeCurrentGiven = TRUE;
            break;
        case DIO_MOD_IKR:
            model->DIOreverseKneeCurrent = value->rValue;
            model->DIOreverseKneeCurrentGiven = TRUE;
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
        case DIO_MOD_FCS:
            model->DIOdepletionSWcapCoeff = value->rValue;
            model->DIOdepletionSWcapCoeffGiven = TRUE;
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
