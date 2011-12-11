/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "txldefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
TXLparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    TXLinstance *here = (TXLinstance *)inst;

    NG_IGNORE(select);

    switch(param) {
        case TXL_IN_NODE:
            here->TXLposNode = value->iValue;
            break;
        case TXL_OUT_NODE:
            here->TXLnegNode = value->iValue;
            break;
		case TXL_LENGTH:
			here->TXLlength = value->rValue;
			here->TXLlengthgiven = TRUE;
			break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
