/**********
Copyright 2003 Paolo Nenzi
Author: 2003 Paolo Nenzi
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "inddefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
INDmParam(int param, IFvalue *value, GENmodel *inModel)
{
    INDmodel *mod = (INDmodel*)inModel;
    switch(param) {
    case IND_MOD_IND:
        mod->INDmInd = value->rValue;
        mod->INDmIndGiven = TRUE;
        break;
    case IND_MOD_TNOM:
        mod->INDtnom = value->rValue+CONSTCtoK;
        mod->INDtnomGiven = TRUE;
        break;
    case IND_MOD_TC1:
        mod->INDtempCoeff1 = value->rValue;
        mod->INDtc1Given = TRUE;
        break;
    case IND_MOD_TC2:
        mod->INDtempCoeff2 = value->rValue;
        mod->INDtc2Given = TRUE;
        break;
    case IND_MOD_CSECT:
        mod->INDcsect = value->rValue;
        mod->INDcsectGiven = TRUE;
        break;
    case IND_MOD_DIA:
        mod->INDdia = value->rValue;
        mod->INDdiaGiven = TRUE;
        break;
    case IND_MOD_LENGTH :
        mod->INDlength = value->rValue;
        mod->INDlengthGiven = TRUE;
        break;
    case IND_MOD_NT :
        mod->INDmodNt = value->rValue;
        mod->INDmodNtGiven = TRUE;
        break;
    case IND_MOD_MU:
        mod->INDmu = value->rValue;
        mod->INDmuGiven = TRUE;
        break;
    case IND_MOD_L:
        /* just being reassured by the user that we are an inductor */
        /* no-op */
        break;
    default:
        return(E_BADPARM);
    }
    return(OK);
}

