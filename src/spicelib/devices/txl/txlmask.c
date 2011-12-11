/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/


#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "txldefs.h"
#include "ngspice/sperror.h"
#include "ngspice/devdefs.h"
#include "ngspice/suffix.h"
#include "ngspice/swec.h"


/* ARGSUSED */
int 
TXLmodAsk(CKTcircuit *ckt, GENmodel *inModel, int which, IFvalue *value)
{
    TXLmodel *model = (TXLmodel *)inModel;

    NG_IGNORE(ckt);

    switch(which) {
        case TXL_R:
            value->rValue = model->R;
            return(OK);
        case TXL_C:
            value->rValue = model->C;
            return(OK);
        case TXL_G:
            value->rValue = model->G;
            return(OK);
        case TXL_L:
            value->rValue = model->L;
            return(OK);
        case TXL_length:
            value->rValue = model->length;
            return(OK);
        default:
            return(E_BADPARM);
    }
}

