/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/


#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "txldefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
TXLask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, IFvalue *select)
{
    TXLinstance *fast = (TXLinstance *)inst;

    NG_IGNORE(select);
    NG_IGNORE(ckt);

    switch(which) {
        case TXL_OUT_NODE:
            value->iValue = fast->TXLnegNode;
            return(OK);
        case TXL_IN_NODE:
            value->iValue = fast->TXLposNode;
            return(OK);
        case TXL_LENGTH:
            value->rValue = fast->TXLlength;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}
