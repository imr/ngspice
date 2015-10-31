/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi
Modified: 2000 AlansFixes
**********/


#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "resdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"


int 
RESmodAsk(CKTcircuit *ckt, GENmodel *inModel, int which, IFvalue *value)
{

    RESmodel *model = (RESmodel *)inModel;

    NG_IGNORE(ckt);

    switch(which) {
        case RES_MOD_TNOM:
            value->rValue = model->REStnom-CONSTCtoK;
            return(OK);
        case RES_MOD_TC1:
            value->rValue = model->REStempCoeff1;
            return(OK);
        case RES_MOD_TC2:
            value->rValue = model->REStempCoeff2;
            return(OK);
        case RES_MOD_TCE:
            value->rValue = model->REStempCoeffe;
            return(OK);
        case RES_MOD_RSH:
            value->rValue = model->RESsheetRes;
            return(OK);
        case RES_MOD_DEFWIDTH:
            value->rValue = model->RESdefWidth;
            return(OK);
        case RES_MOD_DEFLENGTH:
            value->rValue = model->RESdefLength;
            return(OK);
        case RES_MOD_NARROW: 
            value->rValue = model->RESnarrow;
            return(OK);
        case RES_MOD_SHORT: 
            value->rValue = model->RESshort;
            return(OK);
        case RES_MOD_KF:
            if (model->RESfNcoefGiven)
                value->rValue = model->RESfNcoef;
            else
                value->rValue = 0.0;
            return(OK);
        case RES_MOD_AF:
            if (model->RESfNexpGiven)
                value->rValue = model->RESfNexp;
            else
                value->rValue = 0.0;
            return(OK);
        case RES_MOD_BV_MAX:
            value->rValue = model->RESbv_max;
            return(OK);
        case RES_MOD_R:
            value->rValue = model->RESres;
            return(OK);
        case RES_MOD_LF:
            value->rValue = model->RESlf;
            return(OK);
        case RES_MOD_WF:
            value->rValue = model->RESwf;
            return(OK);
        case RES_MOD_EF:
            value->rValue = model->RESef;
            return(OK);
        default:
            return(E_BADPARM);
    }
}

