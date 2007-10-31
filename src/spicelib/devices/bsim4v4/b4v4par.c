/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4par.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 11/15/2002.
 * Modified by Xuemei Xi, 05/09/2003.
 **********/

#include "ngspice.h"
#include "ifsim.h"
#include "bsim4v4def.h"
#include "sperror.h"
#include "suffix.h"
#include "fteext.h"

int
BSIM4V4param(param,value,inst,select)
int param;
IFvalue *value;
GENinstance *inst;
IFvalue *select;
{
    double scale;

    if ( !cp_getvar( "scale", CP_REAL, (double*) &scale ) ) scale = 1;

    BSIM4V4instance *here = (BSIM4V4instance*)inst;
    switch(param) 
    {   case BSIM4V4_W:
            here->BSIM4V4w = value->rValue*scale;
            here->BSIM4V4wGiven = TRUE;
            break;
        case BSIM4V4_L:
            here->BSIM4V4l = value->rValue*scale;
            here->BSIM4V4lGiven = TRUE;
            break;
        case BSIM4V4_M:
            here->BSIM4V4m = value->rValue;
            here->BSIM4V4mGiven = TRUE;
            break;
        case BSIM4V4_NF:
            here->BSIM4V4nf = value->rValue;
            here->BSIM4V4nfGiven = TRUE;
            break;
        case BSIM4V4_MIN:
            here->BSIM4V4min = value->iValue;
            here->BSIM4V4minGiven = TRUE;
            break;
        case BSIM4V4_AS:
            here->BSIM4V4sourceArea = value->rValue*scale*scale;
            here->BSIM4V4sourceAreaGiven = TRUE;
            break;
        case BSIM4V4_AD:
            here->BSIM4V4drainArea = value->rValue*scale*scale;
            here->BSIM4V4drainAreaGiven = TRUE;
            break;
        case BSIM4V4_PS:
            here->BSIM4V4sourcePerimeter = value->rValue*scale;
            here->BSIM4V4sourcePerimeterGiven = TRUE;
            break;
        case BSIM4V4_PD:
            here->BSIM4V4drainPerimeter = value->rValue*scale;
            here->BSIM4V4drainPerimeterGiven = TRUE;
            break;
        case BSIM4V4_NRS:
            here->BSIM4V4sourceSquares = value->rValue;
            here->BSIM4V4sourceSquaresGiven = TRUE;
            break;
        case BSIM4V4_NRD:
            here->BSIM4V4drainSquares = value->rValue;
            here->BSIM4V4drainSquaresGiven = TRUE;
            break;
        case BSIM4V4_OFF:
            here->BSIM4V4off = value->iValue;
            break;
        case BSIM4V4_SA:
            here->BSIM4V4sa = value->rValue*scale;
            here->BSIM4V4saGiven = TRUE;
            break;
        case BSIM4V4_SB:
            here->BSIM4V4sb = value->rValue*scale;
            here->BSIM4V4sbGiven = TRUE;
            break;
        case BSIM4V4_SD:
            here->BSIM4V4sd = value->rValue*scale;
            here->BSIM4V4sdGiven = TRUE;
            break;
        case BSIM4V4_RBSB:
            here->BSIM4V4rbsb = value->rValue;
            here->BSIM4V4rbsbGiven = TRUE;
            break;
        case BSIM4V4_RBDB:
            here->BSIM4V4rbdb = value->rValue;
            here->BSIM4V4rbdbGiven = TRUE;
            break;
        case BSIM4V4_RBPB:
            here->BSIM4V4rbpb = value->rValue;
            here->BSIM4V4rbpbGiven = TRUE;
            break;
        case BSIM4V4_RBPS:
            here->BSIM4V4rbps = value->rValue;
            here->BSIM4V4rbpsGiven = TRUE;
            break;
        case BSIM4V4_RBPD:
            here->BSIM4V4rbpd = value->rValue;
            here->BSIM4V4rbpdGiven = TRUE;
            break;
        case BSIM4V4_TRNQSMOD:
            here->BSIM4V4trnqsMod = value->iValue;
            here->BSIM4V4trnqsModGiven = TRUE;
            break;
        case BSIM4V4_ACNQSMOD:
            here->BSIM4V4acnqsMod = value->iValue;
            here->BSIM4V4acnqsModGiven = TRUE;
            break;
        case BSIM4V4_RBODYMOD:
            here->BSIM4V4rbodyMod = value->iValue;
            here->BSIM4V4rbodyModGiven = TRUE;
            break;
        case BSIM4V4_RGATEMOD:
            here->BSIM4V4rgateMod = value->iValue;
            here->BSIM4V4rgateModGiven = TRUE;
            break;
        case BSIM4V4_GEOMOD:
            here->BSIM4V4geoMod = value->iValue;
            here->BSIM4V4geoModGiven = TRUE;
            break;
        case BSIM4V4_RGEOMOD:
            here->BSIM4V4rgeoMod = value->iValue;
            here->BSIM4V4rgeoModGiven = TRUE;
            break;
        case BSIM4V4_IC_VDS:
            here->BSIM4V4icVDS = value->rValue;
            here->BSIM4V4icVDSGiven = TRUE;
            break;
        case BSIM4V4_IC_VGS:
            here->BSIM4V4icVGS = value->rValue;
            here->BSIM4V4icVGSGiven = TRUE;
            break;
        case BSIM4V4_IC_VBS:
            here->BSIM4V4icVBS = value->rValue;
            here->BSIM4V4icVBSGiven = TRUE;
            break;
        case BSIM4V4_IC:
            switch(value->v.numValue)
            {   case 3:
                    here->BSIM4V4icVBS = *(value->v.vec.rVec+2);
                    here->BSIM4V4icVBSGiven = TRUE;
                case 2:
                    here->BSIM4V4icVGS = *(value->v.vec.rVec+1);
                    here->BSIM4V4icVGSGiven = TRUE;
                case 1:
                    here->BSIM4V4icVDS = *(value->v.vec.rVec);
                    here->BSIM4V4icVDSGiven = TRUE;
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
