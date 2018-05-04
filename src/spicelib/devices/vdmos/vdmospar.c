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
        case VDMOS_AS:
            here->VDMOSsourceArea = value->rValue * scale * scale;
            here->VDMOSsourceAreaGiven = TRUE;
            break;
        case VDMOS_AD:
            here->VDMOSdrainArea = value->rValue * scale * scale;
            here->VDMOSdrainAreaGiven = TRUE;
            break;
        case VDMOS_PS:
            here->VDMOSsourcePerimiter = value->rValue * scale;
            here->VDMOSsourcePerimiterGiven = TRUE;
            break;
        case VDMOS_PD:
            here->VDMOSdrainPerimiter = value->rValue * scale;
            here->VDMOSdrainPerimiterGiven = TRUE;
            break;
        case VDMOS_NRS:
            here->VDMOSsourceSquares = value->rValue;
            here->VDMOSsourceSquaresGiven = TRUE;
            break;
        case VDMOS_NRD:
            here->VDMOSdrainSquares = value->rValue;
            here->VDMOSdrainSquaresGiven = TRUE;
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
        case VDMOS_L_SENS:
            if(value->iValue) {
                here->VDMOSsenParmNo = 1;
                here->VDMOSsens_l = 1;
            }
            break;
        case VDMOS_W_SENS:
            if(value->iValue) {
                here->VDMOSsenParmNo = 1;
                here->VDMOSsens_w = 1;
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
