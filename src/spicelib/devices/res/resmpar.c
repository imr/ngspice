/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi
Modified: 2000 AlansFixes
**********/

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "resdefs.h"
#include "sperror.h"


int
RESmParam(int param, IFvalue *value, GENmodel *inModel)
{
    RESmodel *model = (RESmodel *)inModel;
    
    switch(param) {
        case RES_MOD_TNOM:
            model->REStnom = value->rValue+CONSTCtoK;
            model->REStnomGiven = TRUE;
            break;
        case RES_MOD_TC1:
            model->REStempCoeff1 = value->rValue;
            model->REStc1Given = TRUE;
            break;
        case RES_MOD_TC2:
            model->REStempCoeff2 = value->rValue;
            model->REStc2Given = TRUE;
            break;
        case RES_MOD_RSH:
            model->RESsheetRes = value->rValue;
            model->RESsheetResGiven = TRUE;
            break;
        case RES_MOD_DEFWIDTH:
            model->RESdefWidth = value->rValue;
            model->RESdefWidthGiven = TRUE;
            break;
        case RES_MOD_NARROW:
            model->RESnarrow = value->rValue;
            model->RESnarrowGiven = TRUE;
            break;
        case RES_MOD_SHORT:
            model->RESshort = value->rValue;
            model->RESshortGiven = TRUE;
            break;    
	case RES_MOD_KF:
	    model->RESfNcoef = value->rValue;
	    model->RESfNcoefGiven = TRUE;
	    break;
	case RES_MOD_AF:
	    model->RESfNexp = value->rValue;
	    model->RESfNexpGiven = TRUE;
	    break;
    
        case RES_MOD_R:
            /* just being reassured by user that this is a resistor model */
            /* no-op */
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
