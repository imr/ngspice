/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Gordon Jacobs
**********/
/*
 */

#include "ngspice.h"
#include "cswdefs.h"
#include "ifsim.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
CSWparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    CSWinstance *here = (CSWinstance*)inst;
    switch(param) {
        case  CSW_CONTROL:
            here->CSWcontName = value->uValue;
            break;
        case CSW_IC_ON:
            if(value->iValue) {
                here->CSWzero_stateGiven = TRUE;
            }
            break;
        case CSW_IC_OFF:
            if(value->iValue) {
                here->CSWzero_stateGiven = FALSE;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
