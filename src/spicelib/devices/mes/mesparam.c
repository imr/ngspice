/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 S. Hwang
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "mesdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
MESparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    MESinstance *here = (MESinstance*)inst;

    NG_IGNORE(select);

    switch (param) {
        case MES_AREA:
            here->MESarea = value->rValue;
            here->MESareaGiven = TRUE;
            break;
        case MES_M:
            here->MESm = value->rValue;
            here->MESmGiven = TRUE;
            break;
        case MES_IC_VDS:
            here->MESicVDS = value->rValue;
            here->MESicVDSGiven = TRUE;
            break;
        case MES_IC_VGS:
            here->MESicVGS = value->rValue;
            here->MESicVGSGiven = TRUE;
            break;
        case MES_OFF:
            here->MESoff = value->iValue;
            break;
        case MES_IC:
            /* FALLTHROUGH added to suppress GCC warning due to
             * -Wimplicit-fallthrough flag */
            switch (value->v.numValue) {
                case 2:
                    here->MESicVGS = *(value->v.vec.rVec+1);
                    here->MESicVGSGiven = TRUE;
                    /* FALLTHROUGH */
                case 1:
                    here->MESicVDS = *(value->v.vec.rVec);
                    here->MESicVDSGiven = TRUE;
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
