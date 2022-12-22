/**** BSIM4.6.2 Released by Wenwei Yang 07/31/2008 ****/

/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4par.c of BSIM4.6.2.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, Mohan Dunga, 07/29/2005.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "bsim4v6def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/fteext.h"

int
BSIM4v6param(
int param,
IFvalue *value,
GENinstance *inst,
IFvalue *select)
{
    double scale;

    BSIM4v6instance *here = (BSIM4v6instance*)inst;

    NG_IGNORE(select);

    if (!cp_getvar("scale", CP_REAL, &scale, 0))
        scale = 1;

    switch (param) {
        case BSIM4v6_W:
            here->BSIM4v6w = value->rValue*scale;
            here->BSIM4v6wGiven = TRUE;
            break;
        case BSIM4v6_L:
            here->BSIM4v6l = value->rValue*scale;
            here->BSIM4v6lGiven = TRUE;
            break;
        case BSIM4v6_M:
            here->BSIM4v6m = value->rValue;
            here->BSIM4v6mGiven = TRUE;
            break;
        case BSIM4v6_NF:
            here->BSIM4v6nf = value->rValue;
            here->BSIM4v6nfGiven = TRUE;
            break;
        case BSIM4v6_MIN:
            here->BSIM4v6min = value->iValue;
            here->BSIM4v6minGiven = TRUE;
            break;
        case BSIM4v6_AS:
            here->BSIM4v6sourceArea = value->rValue*scale*scale;
            here->BSIM4v6sourceAreaGiven = TRUE;
            break;
        case BSIM4v6_AD:
            here->BSIM4v6drainArea = value->rValue*scale*scale;
            here->BSIM4v6drainAreaGiven = TRUE;
            break;
        case BSIM4v6_PS:
            here->BSIM4v6sourcePerimeter = value->rValue*scale;
            here->BSIM4v6sourcePerimeterGiven = TRUE;
            break;
        case BSIM4v6_PD:
            here->BSIM4v6drainPerimeter = value->rValue*scale;
            here->BSIM4v6drainPerimeterGiven = TRUE;
            break;
        case BSIM4v6_NRS:
            here->BSIM4v6sourceSquares = value->rValue;
            here->BSIM4v6sourceSquaresGiven = TRUE;
            break;
        case BSIM4v6_NRD:
            here->BSIM4v6drainSquares = value->rValue;
            here->BSIM4v6drainSquaresGiven = TRUE;
            break;
        case BSIM4v6_OFF:
            here->BSIM4v6off = value->iValue;
            break;
        case BSIM4v6_SA:
            here->BSIM4v6sa = value->rValue;
            here->BSIM4v6saGiven = TRUE;
            break;
        case BSIM4v6_SB:
            here->BSIM4v6sb = value->rValue;
            here->BSIM4v6sbGiven = TRUE;
            break;
        case BSIM4v6_SD:
            here->BSIM4v6sd = value->rValue;
            here->BSIM4v6sdGiven = TRUE;
            break;
        case BSIM4v6_SCA:
            here->BSIM4v6sca = value->rValue;
            here->BSIM4v6scaGiven = TRUE;
            break;
        case BSIM4v6_SCB:
            here->BSIM4v6scb = value->rValue;
            here->BSIM4v6scbGiven = TRUE;
            break;
        case BSIM4v6_SCC:
            here->BSIM4v6scc = value->rValue;
            here->BSIM4v6sccGiven = TRUE;
            break;
        case BSIM4v6_SC:
            here->BSIM4v6sc = value->rValue;
            here->BSIM4v6scGiven = TRUE;
            break;
        case BSIM4v6_RBSB:
            here->BSIM4v6rbsb = value->rValue;
            here->BSIM4v6rbsbGiven = TRUE;
            break;
        case BSIM4v6_RBDB:
            here->BSIM4v6rbdb = value->rValue;
            here->BSIM4v6rbdbGiven = TRUE;
            break;
        case BSIM4v6_RBPB:
            here->BSIM4v6rbpb = value->rValue;
            here->BSIM4v6rbpbGiven = TRUE;
            break;
        case BSIM4v6_RBPS:
            here->BSIM4v6rbps = value->rValue;
            here->BSIM4v6rbpsGiven = TRUE;
            break;
        case BSIM4v6_RBPD:
            here->BSIM4v6rbpd = value->rValue;
            here->BSIM4v6rbpdGiven = TRUE;
            break;
        case BSIM4v6_DELVTO:
            here->BSIM4v6delvto = value->rValue;
            here->BSIM4v6delvtoGiven = TRUE;
            break;
        case BSIM4v6_MULU0:
            here->BSIM4v6mulu0 = value->rValue;
            here->BSIM4v6mulu0Given = TRUE;
            break;
        case BSIM4v6_XGW:
            here->BSIM4v6xgw = value->rValue;
            here->BSIM4v6xgwGiven = TRUE;
            break;
        case BSIM4v6_NGCON:
            here->BSIM4v6ngcon = value->rValue;
            here->BSIM4v6ngconGiven = TRUE;
            break;
        case BSIM4v6_TRNQSMOD:
            here->BSIM4v6trnqsMod = value->iValue;
            here->BSIM4v6trnqsModGiven = TRUE;
            break;
        case BSIM4v6_ACNQSMOD:
            here->BSIM4v6acnqsMod = value->iValue;
            here->BSIM4v6acnqsModGiven = TRUE;
            break;
        case BSIM4v6_RBODYMOD:
            here->BSIM4v6rbodyMod = value->iValue;
            here->BSIM4v6rbodyModGiven = TRUE;
            break;
        case BSIM4v6_RGATEMOD:
            here->BSIM4v6rgateMod = value->iValue;
            here->BSIM4v6rgateModGiven = TRUE;
            break;
        case BSIM4v6_GEOMOD:
            here->BSIM4v6geoMod = value->iValue;
            here->BSIM4v6geoModGiven = TRUE;
            break;
        case BSIM4v6_RGEOMOD:
            here->BSIM4v6rgeoMod = value->iValue;
            here->BSIM4v6rgeoModGiven = TRUE;
            break;
        case BSIM4v6_IC_VDS:
            here->BSIM4v6icVDS = value->rValue;
            here->BSIM4v6icVDSGiven = TRUE;
            break;
        case BSIM4v6_IC_VGS:
            here->BSIM4v6icVGS = value->rValue;
            here->BSIM4v6icVGSGiven = TRUE;
            break;
        case BSIM4v6_IC_VBS:
            here->BSIM4v6icVBS = value->rValue;
            here->BSIM4v6icVBSGiven = TRUE;
            break;
        case BSIM4v6_IC:
    /* FALLTHROUGH added to suppress GCC warning due to
     * -Wimplicit-fallthrough flag */
            switch (value->v.numValue) {
                case 3:
                    here->BSIM4v6icVBS = *(value->v.vec.rVec+2);
                    here->BSIM4v6icVBSGiven = TRUE;
                    /* FALLTHROUGH */
                case 2:
                    here->BSIM4v6icVGS = *(value->v.vec.rVec+1);
                    here->BSIM4v6icVGSGiven = TRUE;
                    /* FALLTHROUGH */
                case 1:
                    here->BSIM4v6icVDS = *(value->v.vec.rVec);
                    here->BSIM4v6icVDSGiven = TRUE;
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
