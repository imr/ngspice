/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/29/2005 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4par.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, Mohan Dunga, 07/29/2005.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "bsim4v5def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/fteext.h"

int
BSIM4v5param(
int param,
IFvalue *value,
GENinstance *inst,
IFvalue *select)
{
    double scale;

    BSIM4v5instance *here = (BSIM4v5instance*)inst;

    NG_IGNORE(select);

    if (!cp_getvar("scale", CP_REAL, &scale, 0))
        scale = 1;

    switch (param) {
        case BSIM4v5_W:
            here->BSIM4v5w = value->rValue*scale;
            here->BSIM4v5wGiven = TRUE;
            break;
        case BSIM4v5_L:
            here->BSIM4v5l = value->rValue*scale;
            here->BSIM4v5lGiven = TRUE;
            break;
        case BSIM4v5_M:
            here->BSIM4v5m = value->rValue;
            here->BSIM4v5mGiven = TRUE;
            break;
        case BSIM4v5_NF:
            here->BSIM4v5nf = value->rValue;
            here->BSIM4v5nfGiven = TRUE;
            break;
        case BSIM4v5_MIN:
            here->BSIM4v5min = value->iValue;
            here->BSIM4v5minGiven = TRUE;
            break;
        case BSIM4v5_AS:
            here->BSIM4v5sourceArea = value->rValue*scale*scale;
            here->BSIM4v5sourceAreaGiven = TRUE;
            break;
        case BSIM4v5_AD:
            here->BSIM4v5drainArea = value->rValue*scale*scale;
            here->BSIM4v5drainAreaGiven = TRUE;
            break;
        case BSIM4v5_PS:
            here->BSIM4v5sourcePerimeter = value->rValue*scale;
            here->BSIM4v5sourcePerimeterGiven = TRUE;
            break;
        case BSIM4v5_PD:
            here->BSIM4v5drainPerimeter = value->rValue*scale;
            here->BSIM4v5drainPerimeterGiven = TRUE;
            break;
        case BSIM4v5_NRS:
            here->BSIM4v5sourceSquares = value->rValue;
            here->BSIM4v5sourceSquaresGiven = TRUE;
            break;
        case BSIM4v5_NRD:
            here->BSIM4v5drainSquares = value->rValue;
            here->BSIM4v5drainSquaresGiven = TRUE;
            break;
        case BSIM4v5_OFF:
            here->BSIM4v5off = value->iValue;
            break;
        case BSIM4v5_SA:
            here->BSIM4v5sa = value->rValue*scale;
            here->BSIM4v5saGiven = TRUE;
            break;
        case BSIM4v5_SB:
            here->BSIM4v5sb = value->rValue*scale;
            here->BSIM4v5sbGiven = TRUE;
            break;
        case BSIM4v5_SD:
            here->BSIM4v5sd = value->rValue*scale;
            here->BSIM4v5sdGiven = TRUE;
            break;
        case BSIM4v5_SCA:
            here->BSIM4v5sca = value->rValue;
            here->BSIM4v5scaGiven = TRUE;
            break;
        case BSIM4v5_SCB:
            here->BSIM4v5scb = value->rValue;
            here->BSIM4v5scbGiven = TRUE;
            break;
        case BSIM4v5_SCC:
            here->BSIM4v5scc = value->rValue;
            here->BSIM4v5sccGiven = TRUE;
            break;
        case BSIM4v5_SC:
            here->BSIM4v5sc = value->rValue*scale;
            here->BSIM4v5scGiven = TRUE;
            break;
        case BSIM4v5_RBSB:
            here->BSIM4v5rbsb = value->rValue;
            here->BSIM4v5rbsbGiven = TRUE;
            break;
        case BSIM4v5_RBDB:
            here->BSIM4v5rbdb = value->rValue;
            here->BSIM4v5rbdbGiven = TRUE;
            break;
        case BSIM4v5_RBPB:
            here->BSIM4v5rbpb = value->rValue;
            here->BSIM4v5rbpbGiven = TRUE;
            break;
        case BSIM4v5_RBPS:
            here->BSIM4v5rbps = value->rValue;
            here->BSIM4v5rbpsGiven = TRUE;
            break;
        case BSIM4v5_RBPD:
            here->BSIM4v5rbpd = value->rValue;
            here->BSIM4v5rbpdGiven = TRUE;
            break;
        case BSIM4v5_DELVTO:
            here->BSIM4v5delvto = value->rValue;
            here->BSIM4v5delvtoGiven = TRUE;
            break;
        case BSIM4v5_MULU0:
            here->BSIM4v5mulu0 = value->rValue;
            here->BSIM4v5mulu0Given = TRUE;
            break;
        case BSIM4v5_XGW:
            here->BSIM4v5xgw = value->rValue;
            here->BSIM4v5xgwGiven = TRUE;
            break;
        case BSIM4v5_NGCON:
            here->BSIM4v5ngcon = value->rValue;
            here->BSIM4v5ngconGiven = TRUE;
            break;
        case BSIM4v5_TRNQSMOD:
            here->BSIM4v5trnqsMod = value->iValue;
            here->BSIM4v5trnqsModGiven = TRUE;
            break;
        case BSIM4v5_ACNQSMOD:
            here->BSIM4v5acnqsMod = value->iValue;
            here->BSIM4v5acnqsModGiven = TRUE;
            break;
        case BSIM4v5_RBODYMOD:
            here->BSIM4v5rbodyMod = value->iValue;
            here->BSIM4v5rbodyModGiven = TRUE;
            break;
        case BSIM4v5_RGATEMOD:
            here->BSIM4v5rgateMod = value->iValue;
            here->BSIM4v5rgateModGiven = TRUE;
            break;
        case BSIM4v5_GEOMOD:
            here->BSIM4v5geoMod = value->iValue;
            here->BSIM4v5geoModGiven = TRUE;
            break;
        case BSIM4v5_RGEOMOD:
            here->BSIM4v5rgeoMod = value->iValue;
            here->BSIM4v5rgeoModGiven = TRUE;
            break;
        case BSIM4v5_IC_VDS:
            here->BSIM4v5icVDS = value->rValue;
            here->BSIM4v5icVDSGiven = TRUE;
            break;
        case BSIM4v5_IC_VGS:
            here->BSIM4v5icVGS = value->rValue;
            here->BSIM4v5icVGSGiven = TRUE;
            break;
        case BSIM4v5_IC_VBS:
            here->BSIM4v5icVBS = value->rValue;
            here->BSIM4v5icVBSGiven = TRUE;
            break;
        case BSIM4v5_IC:
            /* FALLTHROUGH added to suppress GCC warning due to
             * -Wimplicit-fallthrough flag */
            switch (value->v.numValue) {
                case 3:
                    here->BSIM4v5icVBS = *(value->v.vec.rVec+2);
                    here->BSIM4v5icVBSGiven = TRUE;
                    /* FALLTHROUGH */
                case 2:
                    here->BSIM4v5icVGS = *(value->v.vec.rVec+1);
                    here->BSIM4v5icVGSGiven = TRUE;
                    /* FALLTHROUGH */
                case 1:
                    here->BSIM4v5icVDS = *(value->v.vec.rVec);
                    here->BSIM4v5icVDSGiven = TRUE;
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
