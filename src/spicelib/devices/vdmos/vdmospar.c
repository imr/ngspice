/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "vdmosdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/fteext.h"


/* ARGSUSED */
int
VDMOSparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    double scale;

    VDMOSinstance *here = (VDMOSinstance *)inst;

    NG_IGNORE(select);

    if (!cp_getvar("scale", CP_REAL, &scale))
        scale = 1;

    switch(param) {
        case VDMOS_TEMP:
            here->VDMOStemp = value->rValue+CONSTCtoK;
            here->VDMOStempGiven = TRUE;
            break;
        case VDMOS_DTEMP:
            here->VDMOSdtemp = value->rValue;
            here->VDMOSdtempGiven = TRUE;
            break;
        case VDMOS_M:
            here->VDMOSm = value->rValue;
            here->VDMOSmGiven = TRUE;
            break;
        case VDMOS_W:
            here->VDMOSw = value->rValue * scale;
            here->VDMOSwGiven = TRUE;
            break;
        case VDMOS_L:
            here->VDMOSl = value->rValue * scale;
            here->VDMOSlGiven = TRUE;
            break;
        case VDMOS_OFF:
            here->VDMOSoff = (value->iValue != 0);
            break;
        case VDMOS_IC_VBS:
            here->VDMOSicVBS = value->rValue;
            here->VDMOSicVBSGiven = TRUE;
            break;
        case VDMOS_IC_VDS:
            here->VDMOSicVDS = value->rValue;
            here->VDMOSicVDSGiven = TRUE;
            break;
        case VDMOS_IC_VGS:
            here->VDMOSicVGS = value->rValue;
            here->VDMOSicVGSGiven = TRUE;
            break;
        case VDMOS_IC:
            switch(value->v.numValue){
                case 3:
                    here->VDMOSicVBS = *(value->v.vec.rVec+2);
                    here->VDMOSicVBSGiven = TRUE;
                case 2:
                    here->VDMOSicVGS = *(value->v.vec.rVec+1);
                    here->VDMOSicVGSGiven = TRUE;
                case 1:
                    here->VDMOSicVDS = *(value->v.vec.rVec);
                    here->VDMOSicVDSGiven = TRUE;
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
