/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

/*
 * This routine sets instance parameters for
 * BJTs in the circuit.
 */

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "bjtdefs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
BJTparam(int param, IFvalue *value, GENinstance *instPtr, IFvalue *select)
{
    BJTinstance *here = (BJTinstance*)instPtr;

    switch(param) {
        case BJT_AREA:
            here->BJTarea = value->rValue;
            here->BJTareaGiven = TRUE;
            break;
	case BJT_AREAB:
            here->BJTareab = value->rValue;
            here->BJTareabGiven = TRUE;
            break;   
	case BJT_AREAC:
            here->BJTareac = value->rValue;
            here->BJTareacGiven = TRUE;
            break;     
        case BJT_M:
            here->BJTm = value->rValue;
            here->BJTmGiven = TRUE;
            break;	    
        case BJT_TEMP:
            here->BJTtemp = value->rValue + CONSTCtoK;
            here->BJTtempGiven = TRUE;
            break;
        case BJT_DTEMP:
            here->BJTdtemp = value->rValue;
            here->BJTdtempGiven = TRUE;
            break;	    
        case BJT_OFF:
            here->BJToff = value->iValue;
            break;
        case BJT_IC_VBE:
            here->BJTicVBE = value->rValue;
            here->BJTicVBEGiven = TRUE;
            break;
        case BJT_IC_VCE:
            here->BJTicVCE = value->rValue;
            here->BJTicVCEGiven = TRUE;
            break;
        case BJT_AREA_SENS:
            here->BJTsenParmNo = value->iValue;
            break;
        case BJT_IC :
            switch(value->v.numValue) {
                case 2:
                    here->BJTicVCE = *(value->v.vec.rVec+1);
                    here->BJTicVCEGiven = TRUE;
                case 1:
                    here->BJTicVBE = *(value->v.vec.rVec);
                    here->BJTicVBEGiven = TRUE;
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
