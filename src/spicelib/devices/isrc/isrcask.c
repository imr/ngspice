/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
**********/
/*
 */

/*
 * This routine gives access to the internal device parameters
 * of independent current SouRCe
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "ifsim.h"
#include "isrcdefs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
ISRCask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, IFvalue *select)
{
    ISRCinstance *here = (ISRCinstance*)inst;
    static char *msg = "Current and power not available in ac analysis";
    int temp;
    double *v, *w;

    switch(which) {
        case ISRC_DC:
            value->rValue = here->ISRCdcValue;
            return (OK);
        case ISRC_AC_MAG:
            value->rValue = here->ISRCacMag;
            return (OK);
        case ISRC_AC_PHASE:
            value->rValue = here->ISRCacPhase;
            return (OK);
        case ISRC_PULSE:
        case ISRC_SINE:
        case ISRC_EXP:
        case ISRC_PWL:
        case ISRC_SFFM:
	case ISRC_AM:
        case ISRC_FCN_COEFFS:
            temp = value->v.numValue = here->ISRCfunctionOrder;
            value->v.vec.rVec = (double *) 
                    MALLOC(here->ISRCfunctionOrder * sizeof(double));
	    v = value->v.vec.rVec;
	    w = here->ISRCcoeffs;
            while (temp--)
                *v++ = *w++;
            return (OK);
        case ISRC_NEG_NODE:
            value->iValue = here->ISRCnegNode;
            return (OK);
        case ISRC_POS_NODE:
            value->iValue = here->ISRCposNode;
            return (OK);
        case ISRC_FCN_TYPE:
            value->iValue = here->ISRCfunctionType;
        case ISRC_AC_REAL:
            value->rValue = here->ISRCacReal;
            return (OK);
        case ISRC_AC_IMAG:
            value->rValue = here->ISRCacImag;
            return (OK);
        case ISRC_FCN_ORDER:
            value->rValue = here->ISRCfunctionOrder;
            return (OK);
        case ISRC_VOLTS:
	    value->rValue = (*(ckt->CKTrhsOld + here->ISRCposNode) -
		*(ckt->CKTrhsOld + here->ISRCnegNode));
            return(OK);
        case ISRC_POWER:
            if (ckt->CKTcurrentAnalysis & DOING_AC) {
                errMsg = MALLOC(strlen(msg)+1);
                errRtn = "ISRCask";
                strcpy(errMsg,msg);
                return(E_ASKPOWER);
            } else {
                value->rValue = -here->ISRCdcValue * 
                        (*(ckt->CKTrhsOld + here->ISRCposNode) -
                        *(ckt->CKTrhsOld + here->ISRCnegNode));
            }
            return(OK);
/* gtri - begin - add current value information */
#ifdef XSPICE
        case ISRC_CURRENT:
            value->rValue = here->ISRCcurrent;
            return (OK);
#endif	    
/* gtri - end - add current value information */
        default:
            return (E_BADPARM);
    }
    /* NOTREACHED */
}
