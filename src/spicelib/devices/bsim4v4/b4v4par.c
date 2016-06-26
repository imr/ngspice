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

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "bsim4v4def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"
#include "ngspice/fteext.h"

int
BSIM4v4param(
int param,
IFvalue *value,
GENinstance *inst,
IFvalue *select)
{
    BSIM4v4instance *here = (BSIM4v4instance*)inst;

    NG_IGNORE(select);

    switch(param)
    {   case BSIM4v4_W:
            here->BSIM4v4w = value->rValue;
            here->BSIM4v4wGiven = TRUE;
            break;
        case BSIM4v4_L:
            here->BSIM4v4l = value->rValue;
            here->BSIM4v4lGiven = TRUE;
            break;
        case BSIM4v4_NF:
            here->BSIM4v4nf = value->rValue;
            here->BSIM4v4nfGiven = TRUE;
            break;
        case BSIM4v4_MIN:
            here->BSIM4v4min = value->iValue;
            here->BSIM4v4minGiven = TRUE;
            break;
        case BSIM4v4_AS:
            here->BSIM4v4sourceArea = value->rValue;
            here->BSIM4v4sourceAreaGiven = TRUE;
            break;
        case BSIM4v4_AD:
            here->BSIM4v4drainArea = value->rValue;
            here->BSIM4v4drainAreaGiven = TRUE;
            break;
        case BSIM4v4_PS:
            here->BSIM4v4sourcePerimeter = value->rValue;
            here->BSIM4v4sourcePerimeterGiven = TRUE;
            break;
        case BSIM4v4_PD:
            here->BSIM4v4drainPerimeter = value->rValue;
            here->BSIM4v4drainPerimeterGiven = TRUE;
            break;
        case BSIM4v4_NRS:
            here->BSIM4v4sourceSquares = value->rValue;
            here->BSIM4v4sourceSquaresGiven = TRUE;
            break;
        case BSIM4v4_NRD:
            here->BSIM4v4drainSquares = value->rValue;
            here->BSIM4v4drainSquaresGiven = TRUE;
            break;
        case BSIM4v4_OFF:
            here->BSIM4v4off = value->iValue;
            break;
        case BSIM4v4_SA:
            here->BSIM4v4sa = value->rValue;
            here->BSIM4v4saGiven = TRUE;
            break;
        case BSIM4v4_SB:
            here->BSIM4v4sb = value->rValue;
            here->BSIM4v4sbGiven = TRUE;
            break;
        case BSIM4v4_SD:
            here->BSIM4v4sd = value->rValue;
            here->BSIM4v4sdGiven = TRUE;
            break;
        case BSIM4v4_RBSB:
            here->BSIM4v4rbsb = value->rValue;
            here->BSIM4v4rbsbGiven = TRUE;
            break;
        case BSIM4v4_RBDB:
            here->BSIM4v4rbdb = value->rValue;
            here->BSIM4v4rbdbGiven = TRUE;
            break;
        case BSIM4v4_RBPB:
            here->BSIM4v4rbpb = value->rValue;
            here->BSIM4v4rbpbGiven = TRUE;
            break;
        case BSIM4v4_RBPS:
            here->BSIM4v4rbps = value->rValue;
            here->BSIM4v4rbpsGiven = TRUE;
            break;
        case BSIM4v4_RBPD:
            here->BSIM4v4rbpd = value->rValue;
            here->BSIM4v4rbpdGiven = TRUE;
            break;
        case BSIM4v4_TRNQSMOD:
            here->BSIM4v4trnqsMod = value->iValue;
            here->BSIM4v4trnqsModGiven = TRUE;
            break;
        case BSIM4v4_ACNQSMOD:
            here->BSIM4v4acnqsMod = value->iValue;
            here->BSIM4v4acnqsModGiven = TRUE;
            break;
        case BSIM4v4_RBODYMOD:
            here->BSIM4v4rbodyMod = value->iValue;
            here->BSIM4v4rbodyModGiven = TRUE;
            break;
        case BSIM4v4_RGATEMOD:
            here->BSIM4v4rgateMod = value->iValue;
            here->BSIM4v4rgateModGiven = TRUE;
            break;
        case BSIM4v4_GEOMOD:
            here->BSIM4v4geoMod = value->iValue;
            here->BSIM4v4geoModGiven = TRUE;
            break;
        case BSIM4v4_RGEOMOD:
            here->BSIM4v4rgeoMod = value->iValue;
            here->BSIM4v4rgeoModGiven = TRUE;
            break;
        case BSIM4v4_IC_VDS:
            here->BSIM4v4icVDS = value->rValue;
            here->BSIM4v4icVDSGiven = TRUE;
            break;
        case BSIM4v4_IC_VGS:
            here->BSIM4v4icVGS = value->rValue;
            here->BSIM4v4icVGSGiven = TRUE;
            break;
        case BSIM4v4_IC_VBS:
            here->BSIM4v4icVBS = value->rValue;
            here->BSIM4v4icVBSGiven = TRUE;
            break;
        case BSIM4v4_IC:
            switch(value->v.numValue)
            {   case 3:
                    here->BSIM4v4icVBS = *(value->v.vec.rVec+2);
                    here->BSIM4v4icVBSGiven = TRUE;
                case 2:
                    here->BSIM4v4icVGS = *(value->v.vec.rVec+1);
                    here->BSIM4v4icVGSGiven = TRUE;
                case 1:
                    here->BSIM4v4icVDS = *(value->v.vec.rVec);
                    here->BSIM4v4icVDSGiven = TRUE;
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
