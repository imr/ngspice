/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: Alan Gillespie
**********/
/*
 */

/*
 * This routine sets instance parameters for
 * BJT2s in the circuit.
 */

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "bjt2defs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
BJT2param(int param, IFvalue *value, GENinstance *instPtr, IFvalue *select)
{
    BJT2instance *here = (BJT2instance*)instPtr;

    switch(param) {
        case BJT2_AREA:
            here->BJT2area = value->rValue;
            here->BJT2areaGiven = TRUE;
            break;
        case BJT2_AREAB:
            here->BJT2areab = value->rValue;
            here->BJT2areabGiven = TRUE;
            break;   
        case BJT2_AREAC:
            here->BJT2areac = value->rValue;
            here->BJT2areacGiven = TRUE;
            break;         
        case BJT2_M:
            here->BJT2m = value->rValue;
            here->BJT2mGiven = TRUE;
            break;	    
        case BJT2_TEMP:
            here->BJT2temp = value->rValue + CONSTCtoK;
            here->BJT2tempGiven = TRUE;
            break;
        case BJT2_DTEMP:
            here->BJT2dtemp = value->rValue;
            here->BJT2dtempGiven = TRUE;
            break;	    
        case BJT2_OFF:
            here->BJT2off = value->iValue;
            break;
        case BJT2_IC_VBE:
            here->BJT2icVBE = value->rValue;
            here->BJT2icVBEGiven = TRUE;
            break;
        case BJT2_IC_VCE:
            here->BJT2icVCE = value->rValue;
            here->BJT2icVCEGiven = TRUE;
            break;
        case BJT2_AREA_SENS:
            here->BJT2senParmNo = value->iValue;
            break;
        case BJT2_IC :
            switch(value->v.numValue) {
                case 2:
                    here->BJT2icVCE = *(value->v.vec.rVec+1);
                    here->BJT2icVCEGiven = TRUE;
                case 1:
                    here->BJT2icVBE = *(value->v.vec.rVec);
                    here->BJT2icVBEGiven = TRUE;
                    break;
                default:
                    return(E_BADPARM);
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
