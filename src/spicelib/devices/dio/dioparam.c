/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "diodefs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
DIOparam(param,value,inst,select)
    int param;
    IFvalue *value;
    GENinstance *inst;
    IFvalue *select;
{
    DIOinstance *here = (DIOinstance*)inst;
    switch(param) {
        case DIO_AREA:
            here->DIOarea = value->rValue;
            here->DIOareaGiven = TRUE;
            break;
        case DIO_TEMP:
            here->DIOtemp = value->rValue+CONSTCtoK;
            here->DIOtempGiven = TRUE;
            break;
        case DIO_OFF:
            here->DIOoff = value->iValue;
            break;
        case DIO_IC:
            here->DIOinitCond = value->rValue;
            break;
        case DIO_AREA_SENS:
            here->DIOsenParmNo = value->iValue;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
