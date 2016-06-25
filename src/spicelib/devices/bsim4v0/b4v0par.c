/**** BSIM4.0.0, Released by Weidong Liu 3/24/2000 ****/

/**********
 * Copyright 2000 Regents of the University of California. All rights reserved.
 * File: b4par.c of BSIM4.0.0.
 * Authors: Weidong Liu, Kanyu M. Cao, Xiaodong Jin, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "bsim4v0def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/fteext.h"

int
BSIM4v0param(param,value,inst,select)
int param;
IFvalue *value;
GENinstance *inst;
IFvalue *select;
{
    BSIM4v0instance *here = (BSIM4v0instance*)inst;
    switch(param) 
    {   case BSIM4v0_W:
            here->BSIM4v0w = value->rValue;
            here->BSIM4v0wGiven = TRUE;
            break;
        case BSIM4v0_L:
            here->BSIM4v0l = value->rValue;
            here->BSIM4v0lGiven = TRUE;
            break;
        case BSIM4v0_NF:
            here->BSIM4v0nf = value->rValue;
            here->BSIM4v0nfGiven = TRUE;
            break;
        case BSIM4v0_MIN:
            here->BSIM4v0min = value->iValue;
            here->BSIM4v0minGiven = TRUE;
            break;
        case BSIM4v0_AS:
            here->BSIM4v0sourceArea = value->rValue;
            here->BSIM4v0sourceAreaGiven = TRUE;
            break;
        case BSIM4v0_AD:
            here->BSIM4v0drainArea = value->rValue;
            here->BSIM4v0drainAreaGiven = TRUE;
            break;
        case BSIM4v0_PS:
            here->BSIM4v0sourcePerimeter = value->rValue;
            here->BSIM4v0sourcePerimeterGiven = TRUE;
            break;
        case BSIM4v0_PD:
            here->BSIM4v0drainPerimeter = value->rValue;
            here->BSIM4v0drainPerimeterGiven = TRUE;
            break;
        case BSIM4v0_NRS:
            here->BSIM4v0sourceSquares = value->rValue;
            here->BSIM4v0sourceSquaresGiven = TRUE;
            break;
        case BSIM4v0_NRD:
            here->BSIM4v0drainSquares = value->rValue;
            here->BSIM4v0drainSquaresGiven = TRUE;
            break;
        case BSIM4v0_OFF:
            here->BSIM4v0off = value->iValue;
            break;
        case BSIM4v0_RBSB:
            here->BSIM4v0rbsb = value->rValue;
            here->BSIM4v0rbsbGiven = TRUE;
            break;
        case BSIM4v0_RBDB:
            here->BSIM4v0rbdb = value->rValue;
            here->BSIM4v0rbdbGiven = TRUE;
            break;
        case BSIM4v0_RBPB:
            here->BSIM4v0rbpb = value->rValue;
            here->BSIM4v0rbpbGiven = TRUE;
            break;
        case BSIM4v0_RBPS:
            here->BSIM4v0rbps = value->rValue;
            here->BSIM4v0rbpsGiven = TRUE;
            break;
        case BSIM4v0_RBPD:
            here->BSIM4v0rbpd = value->rValue;
            here->BSIM4v0rbpdGiven = TRUE;
            break;
        case BSIM4v0_TRNQSMOD:
            here->BSIM4v0trnqsMod = value->iValue;
            here->BSIM4v0trnqsModGiven = TRUE;
            break;
        case BSIM4v0_ACNQSMOD:
            here->BSIM4v0acnqsMod = value->iValue;
            here->BSIM4v0acnqsModGiven = TRUE;
            break;
        case BSIM4v0_RBODYMOD:
            here->BSIM4v0rbodyMod = value->iValue;
            here->BSIM4v0rbodyModGiven = TRUE;
            break;
        case BSIM4v0_RGATEMOD:
            here->BSIM4v0rgateMod = value->iValue;
            here->BSIM4v0rgateModGiven = TRUE;
            break;
        case BSIM4v0_GEOMOD:
            here->BSIM4v0geoMod = value->iValue;
            here->BSIM4v0geoModGiven = TRUE;
            break;
        case BSIM4v0_RGEOMOD:
            here->BSIM4v0rgeoMod = value->iValue;
            here->BSIM4v0rgeoModGiven = TRUE;
            break;
        case BSIM4v0_IC_VDS:
            here->BSIM4v0icVDS = value->rValue;
            here->BSIM4v0icVDSGiven = TRUE;
            break;
        case BSIM4v0_IC_VGS:
            here->BSIM4v0icVGS = value->rValue;
            here->BSIM4v0icVGSGiven = TRUE;
            break;
        case BSIM4v0_IC_VBS:
            here->BSIM4v0icVBS = value->rValue;
            here->BSIM4v0icVBSGiven = TRUE;
            break;
        case BSIM4v0_IC:
            switch(value->v.numValue)
            {   case 3:
                    here->BSIM4v0icVBS = *(value->v.vec.rVec+2);
                    here->BSIM4v0icVBSGiven = TRUE;
                case 2:
                    here->BSIM4v0icVGS = *(value->v.vec.rVec+1);
                    here->BSIM4v0icVGSGiven = TRUE;
                case 1:
                    here->BSIM4v0icVDS = *(value->v.vec.rVec);
                    here->BSIM4v0icVDSGiven = TRUE;
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
