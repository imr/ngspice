/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi
Modified: 2000 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "resdefs.h"
#include "ngspice/sperror.h"


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
    case RES_MOD_TCE:
        model->REStempCoeffe = value->rValue;
        model->REStceGiven = TRUE;
        break;
    case RES_MOD_RSH:
        model->RESsheetRes = value->rValue;
        model->RESsheetResGiven = TRUE;
        break;
    case RES_MOD_DEFWIDTH:
        model->RESdefWidth = value->rValue;
        model->RESdefWidthGiven = TRUE;
        break;
    case RES_MOD_DEFLENGTH:
        model->RESdefLength = value->rValue;
        model->RESdefLengthGiven = TRUE;
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
    case RES_MOD_BV_MAX:
        model->RESbv_max = value->rValue;
        model->RESbv_maxGiven = TRUE;
        break;
    case RES_MOD_LF:
        model->RESlf = value->rValue;
        model->RESlfGiven = TRUE;
        break;
    case RES_MOD_WF:
        model->RESwf = value->rValue;
        model->RESwfGiven = TRUE;
        break;
    case RES_MOD_EF:
        model->RESef = value->rValue;
        model->RESefGiven = TRUE;
        break;
    case RES_MOD_R:
        if ( value->rValue > 1e-03 ) {
            model->RESres = value->rValue;
            model->RESresGiven = TRUE;
        }
        break;
    default:
        return(E_BADPARM);
    }
    return(OK);
}
