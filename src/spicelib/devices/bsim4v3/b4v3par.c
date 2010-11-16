/**** BSIM4.3.0 Released by Xuemei(Jane) Xi 05/09/2003 ****/

/**********
 * Copyright 2003 Regents of the University of California. All rights reserved.
 * File: b4v3par.c of BSIM4.3.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 **********/

#include "ngspice.h"
#include "ifsim.h"
#include "bsim4v3def.h"
#include "sperror.h"
#include "fteext.h"

int
BSIM4v3param(
int param,
IFvalue *value,
GENinstance *inst,
IFvalue *select)
{
    double scale;

    BSIM4v3instance *here = (BSIM4v3instance*)inst;

    IGNORE(select);

    if ( !cp_getvar( "scale", CP_REAL, (double*) &scale ) ) scale = 1;

    switch(param) 
    {   case BSIM4v3_W:
            here->BSIM4v3w = value->rValue*scale;
            here->BSIM4v3wGiven = TRUE;
            break;
        case BSIM4v3_L:
            here->BSIM4v3l = value->rValue*scale;
            here->BSIM4v3lGiven = TRUE;
            break;
        case BSIM4v3_M:
            here->BSIM4v3m = value->rValue;
            here->BSIM4v3mGiven = TRUE;
            break;
        case BSIM4v3_NF:
            here->BSIM4v3nf = value->rValue;
            here->BSIM4v3nfGiven = TRUE;
            break;
        case BSIM4v3_MIN:
            here->BSIM4v3min = value->iValue;
            here->BSIM4v3minGiven = TRUE;
            break;
        case BSIM4v3_AS:
            here->BSIM4v3sourceArea = value->rValue*scale*scale;
            here->BSIM4v3sourceAreaGiven = TRUE;
            break;
        case BSIM4v3_AD:
            here->BSIM4v3drainArea = value->rValue*scale*scale;
            here->BSIM4v3drainAreaGiven = TRUE;
            break;
        case BSIM4v3_PS:
            here->BSIM4v3sourcePerimeter = value->rValue*scale;
            here->BSIM4v3sourcePerimeterGiven = TRUE;
            break;
        case BSIM4v3_PD:
            here->BSIM4v3drainPerimeter = value->rValue*scale;
            here->BSIM4v3drainPerimeterGiven = TRUE;
            break;
        case BSIM4v3_NRS:
            here->BSIM4v3sourceSquares = value->rValue;
            here->BSIM4v3sourceSquaresGiven = TRUE;
            break;
        case BSIM4v3_NRD:
            here->BSIM4v3drainSquares = value->rValue;
            here->BSIM4v3drainSquaresGiven = TRUE;
            break;
        case BSIM4v3_OFF:
            here->BSIM4v3off = value->iValue;
            break;
        case BSIM4v3_SA:
            here->BSIM4v3sa = value->rValue;
            here->BSIM4v3saGiven = TRUE;
            break;
        case BSIM4v3_SB:
            here->BSIM4v3sb = value->rValue;
            here->BSIM4v3sbGiven = TRUE;
            break;
        case BSIM4v3_SD:
            here->BSIM4v3sd = value->rValue;
            here->BSIM4v3sdGiven = TRUE;
            break;
        case BSIM4v3_RBSB:
            here->BSIM4v3rbsb = value->rValue;
            here->BSIM4v3rbsbGiven = TRUE;
            break;
        case BSIM4v3_RBDB:
            here->BSIM4v3rbdb = value->rValue;
            here->BSIM4v3rbdbGiven = TRUE;
            break;
        case BSIM4v3_RBPB:
            here->BSIM4v3rbpb = value->rValue;
            here->BSIM4v3rbpbGiven = TRUE;
            break;
        case BSIM4v3_RBPS:
            here->BSIM4v3rbps = value->rValue;
            here->BSIM4v3rbpsGiven = TRUE;
            break;
        case BSIM4v3_RBPD:
            here->BSIM4v3rbpd = value->rValue;
            here->BSIM4v3rbpdGiven = TRUE;
            break;
        case BSIM4v3_TRNQSMOD:
            here->BSIM4v3trnqsMod = value->iValue;
            here->BSIM4v3trnqsModGiven = TRUE;
            break;
        case BSIM4v3_ACNQSMOD:
            here->BSIM4v3acnqsMod = value->iValue;
            here->BSIM4v3acnqsModGiven = TRUE;
            break;
        case BSIM4v3_RBODYMOD:
            here->BSIM4v3rbodyMod = value->iValue;
            here->BSIM4v3rbodyModGiven = TRUE;
            break;
        case BSIM4v3_RGATEMOD:
            here->BSIM4v3rgateMod = value->iValue;
            here->BSIM4v3rgateModGiven = TRUE;
            break;
        case BSIM4v3_GEOMOD:
            here->BSIM4v3geoMod = value->iValue;
            here->BSIM4v3geoModGiven = TRUE;
            break;
        case BSIM4v3_RGEOMOD:
            here->BSIM4v3rgeoMod = value->iValue;
            here->BSIM4v3rgeoModGiven = TRUE;
            break;
        case BSIM4v3_IC_VDS:
            here->BSIM4v3icVDS = value->rValue;
            here->BSIM4v3icVDSGiven = TRUE;
            break;
        case BSIM4v3_IC_VGS:
            here->BSIM4v3icVGS = value->rValue;
            here->BSIM4v3icVGSGiven = TRUE;
            break;
        case BSIM4v3_IC_VBS:
            here->BSIM4v3icVBS = value->rValue;
            here->BSIM4v3icVBSGiven = TRUE;
            break;
        case BSIM4v3_IC:
            switch(value->v.numValue)
            {   case 3:
                    here->BSIM4v3icVBS = *(value->v.vec.rVec+2);
                    here->BSIM4v3icVBSGiven = TRUE;
                case 2:
                    here->BSIM4v3icVGS = *(value->v.vec.rVec+1);
                    here->BSIM4v3icVGSGiven = TRUE;
                case 1:
                    here->BSIM4v3icVDS = *(value->v.vec.rVec);
                    here->BSIM4v3icVDSGiven = TRUE;
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
