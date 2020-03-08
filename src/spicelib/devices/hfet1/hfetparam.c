/**********
Imported from MacSpice3f4 - Antony Wilson
Modified: Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "hfetdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
HFETAparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    HFETAinstance *here = (HFETAinstance*)inst;

    NG_IGNORE(select);

    switch (param) {
        case HFETA_LENGTH:
            here->HFETAlength = value->rValue;
            here->HFETAlengthGiven = TRUE;
            break;
        case HFETA_WIDTH:
            here->HFETAwidth = value->rValue;
            here->HFETAwidthGiven = TRUE;
            break;
        case HFETA_M:
            here->HFETAm = value->rValue;
            here->HFETAmGiven = TRUE;
            break;
        case HFETA_IC_VDS:
            here->HFETAicVDS = value->rValue;
            here->HFETAicVDSGiven = TRUE;
            break;
        case HFETA_IC_VGS:
            here->HFETAicVGS = value->rValue;
            here->HFETAicVGSGiven = TRUE;
            break;
        case HFETA_OFF:
            here->HFETAoff = value->iValue;
            break;
        case HFETA_IC:
            /* FALLTHROUGH added to suppress GCC warning due to
             * -Wimplicit-fallthrough flag */
            switch (value->v.numValue) {
                case 2:
                    here->HFETAicVGS = *(value->v.vec.rVec+1);
                    here->HFETAicVGSGiven = TRUE;
                    /* FALLTHROUGH */
                case 1:
                    here->HFETAicVDS = *(value->v.vec.rVec);
                    here->HFETAicVDSGiven = TRUE;
                    break;
                default:
                    return(E_BADPARM);
            }
            break;
        case HFETA_TEMP:
            here->HFETAtemp = value->rValue + CONSTCtoK;
            here->HFETAtempGiven = TRUE;
            break;
        case HFETA_DTEMP:
            here->HFETAdtemp = value->rValue;
            here->HFETAdtempGiven = TRUE;
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
