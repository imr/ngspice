/**********
License              : 3-clause BSD
Spice3 Implementation: 2019-2020 Dietmar Warning, Markus Müller, Mario Krattenmacher
Model Author         : 1990 Michael Schröter TU Dresden
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
                case 3:
                    here->HICUMicVCS = *(value->v.vec.rVec+2);
                    here->HICUMicVCSGiven = TRUE;
                    /* fallthrough */
                case 2:
                    here->HICUMicVCE = *(value->v.vec.rVec+1);
                    here->HICUMicVCEGiven = TRUE;
                    /* fallthrough */
                case 1:
                    here->HICUMicVBE = *(value->v.vec.rVec);
                    here->HICUMicVBEGiven = TRUE;
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
