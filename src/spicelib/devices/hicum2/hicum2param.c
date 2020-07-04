/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1990 Michael Schröter TU Dresden
Spice3 Implementation: 2019 Dietmar Warning
**********/

/*
 * This routine sets instance parameters for
 * HICUMs in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "hicum2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
HICUMparam(int param, IFvalue *value, GENinstance *instPtr, IFvalue *select)
{
    HICUMinstance *here = (HICUMinstance*)instPtr;

    NG_IGNORE(select);

    switch(param) {
        case HICUM_AREA:
            here->HICUMarea = value->rValue;
            here->HICUMareaGiven = TRUE;
            break;
        case HICUM_OFF:
            here->HICUMoff = (value->iValue != 0);
            break;
        // case HICUM_IC_VB:
        //     here->HICUMicVB      = value->rValue;
        //     here->HICUMicVBGiven = TRUE;
        //     break;
        // case HICUM_IC_VE:
        //     here->HICUMicVE      = value->rValue;
        //     here->HICUMicVEGiven = TRUE;
        //     break;
        // case HICUM_IC_VC:
        //     here->HICUMicVC      = value->rValue;
        //     here->HICUMicVCGiven = TRUE;
        //     break;
        // case HICUM_IC_VBi:
        //     here->HICUMicVBi      = value->rValue;
        //     here->HICUMicVBiGiven = TRUE;
        //     break;
        // case HICUM_IC_VBp:
        //     here->HICUMicVBp      = value->rValue;
        //     here->HICUMicVBpGiven = TRUE;
        //     break;
        // case HICUM_IC_VEi:
        //     here->HICUMicVEi      = value->rValue;
        //     here->HICUMicVEiGiven = TRUE;
        //     break;
        // case HICUM_IC_VCi:
        //     here->HICUMicVCi      = value->rValue;
        //     here->HICUMicVCiGiven = TRUE;
        //     break;
        // case HICUM_IC_Vt:
        //     here->HICUMicVt      = value->rValue;
        //     here->HICUMicVtGiven = TRUE;
        //     break;
        case HICUM_TEMP:
            here->HICUMtemp = value->rValue+CONSTCtoK;
            here->HICUMtempGiven = TRUE;
            break;
        case HICUM_DTEMP:
            here->HICUMdtemp = value->rValue;
            here->HICUMdtempGiven = TRUE;
            break;
        case HICUM_M:
            here->HICUMm = value->rValue;
            here->HICUMmGiven = TRUE;
            break;
        case HICUM_IC :
            switch(value->v.numValue) {
                case 2: //todo
                    here->HICUMicVC = *(value->v.vec.rVec+1);
                    here->HICUMicVCGiven = TRUE;
                    break;
                case 1:
                    here->HICUMicVB = *(value->v.vec.rVec);
                    here->HICUMicVBGiven = TRUE;
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
