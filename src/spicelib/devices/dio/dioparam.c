/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified by Dietmar Warning 2003 and Paolo Nenzi 2003
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
DIOparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    DIOinstance *here = (DIOinstance*)inst;
    switch(param) {
        case DIO_AREA:
            here->DIOarea = value->rValue;
            here->DIOareaGiven = TRUE;
            break;
        case DIO_PJ:
            here->DIOpj = value->rValue;
            here->DIOpjGiven = TRUE;
            break;
        case DIO_M:
            here->DIOm = value->rValue;
            here->DIOmGiven = TRUE;
            break;

        case DIO_TEMP:
            here->DIOtemp = value->rValue+CONSTCtoK;
            here->DIOtempGiven = TRUE;
            break;
	case DIO_DTEMP:
            here->DIOdtemp = value->rValue;
            here->DIOdtempGiven = TRUE;
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
