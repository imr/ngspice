/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified by Dietmar Warning 2003 and Paolo Nenzi 2003
**********/
/*
 */

#include "ngspice.h"
#include "const.h"
#include "cktdefs.h"
#include "ifsim.h"
#include "diodefs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
DIOmAsk (CKTcircuit *ckt, GENmodel *inModel, int which, IFvalue *value)
{
    DIOmodel *model = (DIOmodel*)inModel;
    switch (which) {
        case DIO_MOD_IS:
            value->rValue = model->DIOsatCur;
            return(OK);
        case DIO_MOD_JSW:
            value->rValue = model->DIOsatSWCur;
            return(OK);

        case DIO_MOD_TNOM:
            value->rValue = model->DIOnomTemp-CONSTCtoK;
            return(OK);
        case DIO_MOD_RS:
            value->rValue = model->DIOresist;
            return(OK);
        case DIO_MOD_TRS:
            value->rValue = model->DIOresistTemp1;
            return(OK);	    
        case DIO_MOD_TRS2:
            value->rValue = model->DIOresistTemp2;
            return(OK);	    	    
        case DIO_MOD_N:
            value->rValue = model->DIOemissionCoeff;
            return(OK);
        case DIO_MOD_TT:
            value->rValue = model->DIOtransitTime;
            return(OK);
        case DIO_MOD_TTT1:
            value->rValue = model->DIOtranTimeTemp1;
            return(OK);	    
        case DIO_MOD_TTT2:
            value->rValue = model->DIOtranTimeTemp2;
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
        case DIO_MOD_TM1:
            value->rValue = model->DIOgradCoeffTemp1;
            return(OK);	 
        case DIO_MOD_TM2:
            value->rValue = model->DIOgradCoeffTemp2;
            return(OK);	    	       
        case DIO_MOD_CJSW:
            value->rValue = model->DIOjunctionSWCap;
            return(OK);
        case DIO_MOD_VJSW:
            value->rValue = model->DIOjunctionSWPot;
            return(OK);
        case DIO_MOD_MJSW:
            value->rValue = model->DIOgradingSWCoeff;
            return(OK);
        case DIO_MOD_IKF:
            value->rValue = model->DIOforwardKneeCurrent;
            return(OK);
        case DIO_MOD_IKR:
            value->rValue = model->DIOreverseKneeCurrent;
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
	case DIO_MOD_FCS:
	    value->rValue = model->DIOdepletionSWcapCoeff;
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

