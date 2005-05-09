/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Apr 2000 - Paolo Nenzi
**********/

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "resdefs.h"
#include "sperror.h"


int
RESparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    RESinstance *here = (RESinstance *)inst;
    switch(param) {
        case RES_TEMP:
            here->REStemp = value->rValue + CONSTCtoK;
            here->REStempGiven = TRUE;
            break;
	case RES_DTEMP:
            here->RESdtemp = value->rValue;
            here->RESdtempGiven = TRUE;
            break;   
        case RES_RESIST:
            here->RESresist = value->rValue;
            here->RESresGiven = TRUE;
            break;
        case RES_ACRESIST:
	    here->RESacResist = value->rValue;
	    here->RESacresGiven = TRUE;
	    break;
	case RES_WIDTH:
            here->RESwidth = value->rValue;
            here->RESwidthGiven = TRUE;
            break;
        case RES_LENGTH:
            here->RESlength = value->rValue;
            here->RESlengthGiven = TRUE;
            break;
	case RES_SCALE:
	    here->RESscale = value->rValue;
	    here->RESscaleGiven = TRUE;
	    break;
        case RES_RESIST_SENS:
            here->RESsenParmNo = value->iValue;
	    break;
	case RES_M:
	    here->RESm = value->rValue;
	    here->RESmGiven = TRUE;
            break;
	case RES_TC1:
	    here->REStc1 = value->rValue;
	    here->REStc1Given = TRUE;
            break;
	case RES_TC2:
	    here->REStc2 = value->rValue;
	    here->REStc2Given = TRUE;
            break;
	case RES_NOISY: 
	    here->RESnoisy = value->iValue;
	    here->RESnoisyGiven = TRUE;
	    break;     
        default:
            return(E_BADPARM);
    }
    return(OK);
}
