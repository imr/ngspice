/**********
Copyright 1992 Regents of the University of California.  All rights
reserved.
Author: 1992 Charles Hough
**********/

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "txldefs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
TXLparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    TXLinstance *here = (TXLinstance *)inst;
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
