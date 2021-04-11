/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified by Paolo Nenzi 2003 and Dietmar Warning 2012
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "diodefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/fteext.h"


/* ARGSUSED */
int
DIOparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    DIOinstance *here = (DIOinstance*)inst;

    NG_IGNORE(select);

    switch(param) {
        case DIO_AREA:
            here->DIOarea = value->rValue;
            here->DIOareaGiven = TRUE;
            break;
        case DIO_PJ:
            here->DIOpj = value->rValue;
            here->DIOpjGiven = TRUE;
            break;
        case DIO_W:
            here->DIOw = value->rValue;
            here->DIOwGiven = TRUE;
            break;
        case DIO_L:
            here->DIOl = value->rValue;
            here->DIOlGiven = TRUE;
            break;
        case DIO_M:
            here->DIOm = value->rValue;
            here->DIOmGiven = TRUE;
            break;
        case DIO_THERMAL:
            here->DIOthermal = (value->iValue != 0);
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
            here->DIOoff = (value->iValue != 0);
            break;
        case DIO_IC:
            here->DIOinitCond = value->rValue;
            break;
        case DIO_AREA_SENS:
            here->DIOsenParmNo = value->iValue;
            break;

        case DIO_LM:
            here->DIOlengthMetal = value->rValue;
            here->DIOlengthMetalGiven = TRUE;
            break;
        case DIO_LP:
            here->DIOlengthPoly = value->rValue;
            here->DIOlengthPolyGiven = TRUE;
            break;
        case DIO_WM:
            here->DIOwidthMetal = value->rValue;
            here->DIOwidthMetalGiven = TRUE;
            break;
        case DIO_WP:
            here->DIOwidthPoly = value->rValue;
            here->DIOwidthPolyGiven = TRUE;
            break;

        default:
            return(E_BADPARM);
    }
    return(OK);
}
