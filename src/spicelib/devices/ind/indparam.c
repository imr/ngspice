/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "ifsim.h"
#include "inddefs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
INDparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    INDinstance *here = (INDinstance*)inst;
    switch(param) {
        case IND_IND:
            here->INDinduct = value->rValue;
	    if (!here->INDmGiven) 
	        here->INDm =1.0;
            here->INDindGiven = TRUE;
            break;    
        case IND_TEMP:
            here->INDtemp = value->rValue + CONSTCtoK;
            here->INDtempGiven = TRUE;
            break;
        case IND_DTEMP:
            here->INDdtemp = value->rValue;
            here->INDdtempGiven = TRUE;
            break;
        case IND_M:
            here->INDm = value->rValue;
            here->INDmGiven = TRUE;
            break;
        case IND_SCALE:
            here->INDscale = value->rValue;
            here->INDscaleGiven = TRUE;
            break;
	case IND_NT:
            here->INDnt = value->rValue;
            here->INDntGiven = TRUE;
            break;	    	    
        case IND_IC:
            here->INDinitCond = value->rValue;
            here->INDicGiven = TRUE;
            break;
        case IND_IND_SENS:
            here->INDsenParmNo = value->iValue;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
