/**** BSIM4.2.1, Released by Xuemei Xi 10/05/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b4par.c of BSIM4.2.1.
 * Author: 2000 Weidong Liu
 * Authors: Xuemei Xi, Kanyu M. Cao, Hui Wan, Mansun Chan, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice.h"
#include "ifsim.h"
#include "bsim4v2def.h"
#include "sperror.h"
#include "fteext.h"

int
BSIM4v2param(param,value,inst,select)
int param;
IFvalue *value;
GENinstance *inst;
IFvalue *select;
{
    double scale;

    if ( !cp_getvar( "scale", CP_REAL, (double*) &scale ) ) scale = 1;

    BSIM4v2instance *here = (BSIM4v2instance*)inst;
    switch(param) 
    {   case BSIM4v2_W:
            here->BSIM4v2w = value->rValue*scale;
            here->BSIM4v2wGiven = TRUE;
            break;
        case BSIM4v2_L:
            here->BSIM4v2l = value->rValue*scale;
            here->BSIM4v2lGiven = TRUE;
            break;
        case BSIM4v2_M:
            here->BSIM4v2m = value->rValue;
            here->BSIM4v2mGiven = TRUE;
            break;
        case BSIM4v2_NF:
            here->BSIM4v2nf = value->rValue;
            here->BSIM4v2nfGiven = TRUE;
            break;
        case BSIM4v2_MIN:
            here->BSIM4v2min = value->iValue;
            here->BSIM4v2minGiven = TRUE;
            break;
        case BSIM4v2_AS:
            here->BSIM4v2sourceArea = value->rValue*scale*scale;
            here->BSIM4v2sourceAreaGiven = TRUE;
            break;
        case BSIM4v2_AD:
            here->BSIM4v2drainArea = value->rValue*scale*scale;
            here->BSIM4v2drainAreaGiven = TRUE;
            break;
        case BSIM4v2_PS:
            here->BSIM4v2sourcePerimeter = value->rValue*scale;
            here->BSIM4v2sourcePerimeterGiven = TRUE;
            break;
        case BSIM4v2_PD:
            here->BSIM4v2drainPerimeter = value->rValue*scale;
            here->BSIM4v2drainPerimeterGiven = TRUE;
            break;
        case BSIM4v2_NRS:
            here->BSIM4v2sourceSquares = value->rValue;
            here->BSIM4v2sourceSquaresGiven = TRUE;
            break;
        case BSIM4v2_NRD:
            here->BSIM4v2drainSquares = value->rValue;
            here->BSIM4v2drainSquaresGiven = TRUE;
            break;
        case BSIM4v2_OFF:
            here->BSIM4v2off = value->iValue;
            break;
        case BSIM4v2_RBSB:
            here->BSIM4v2rbsb = value->rValue;
            here->BSIM4v2rbsbGiven = TRUE;
            break;
        case BSIM4v2_RBDB:
            here->BSIM4v2rbdb = value->rValue;
            here->BSIM4v2rbdbGiven = TRUE;
            break;
        case BSIM4v2_RBPB:
            here->BSIM4v2rbpb = value->rValue;
            here->BSIM4v2rbpbGiven = TRUE;
            break;
        case BSIM4v2_RBPS:
            here->BSIM4v2rbps = value->rValue;
            here->BSIM4v2rbpsGiven = TRUE;
            break;
        case BSIM4v2_RBPD:
            here->BSIM4v2rbpd = value->rValue;
            here->BSIM4v2rbpdGiven = TRUE;
            break;
        case BSIM4v2_TRNQSMOD:
            here->BSIM4v2trnqsMod = value->iValue;
            here->BSIM4v2trnqsModGiven = TRUE;
            break;
        case BSIM4v2_ACNQSMOD:
            here->BSIM4v2acnqsMod = value->iValue;
            here->BSIM4v2acnqsModGiven = TRUE;
            break;
        case BSIM4v2_RBODYMOD:
            here->BSIM4v2rbodyMod = value->iValue;
            here->BSIM4v2rbodyModGiven = TRUE;
            break;
        case BSIM4v2_RGATEMOD:
            here->BSIM4v2rgateMod = value->iValue;
            here->BSIM4v2rgateModGiven = TRUE;
            break;
        case BSIM4v2_GEOMOD:
            here->BSIM4v2geoMod = value->iValue;
            here->BSIM4v2geoModGiven = TRUE;
            break;
        case BSIM4v2_RGEOMOD:
            here->BSIM4v2rgeoMod = value->iValue;
            here->BSIM4v2rgeoModGiven = TRUE;
            break;
        case BSIM4v2_IC_VDS:
            here->BSIM4v2icVDS = value->rValue;
            here->BSIM4v2icVDSGiven = TRUE;
            break;
        case BSIM4v2_IC_VGS:
            here->BSIM4v2icVGS = value->rValue;
            here->BSIM4v2icVGSGiven = TRUE;
            break;
        case BSIM4v2_IC_VBS:
            here->BSIM4v2icVBS = value->rValue;
            here->BSIM4v2icVBSGiven = TRUE;
            break;
        case BSIM4v2_IC:
            switch(value->v.numValue)
            {   case 3:
                    here->BSIM4v2icVBS = *(value->v.vec.rVec+2);
                    here->BSIM4v2icVBSGiven = TRUE;
                case 2:
                    here->BSIM4v2icVGS = *(value->v.vec.rVec+1);
                    here->BSIM4v2icVGSGiven = TRUE;
                case 1:
                    here->BSIM4v2icVDS = *(value->v.vec.rVec);
                    here->BSIM4v2icVDSGiven = TRUE;
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
