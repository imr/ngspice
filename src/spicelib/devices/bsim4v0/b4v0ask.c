/**** BSIM4.0.0, Released by Weidong Liu 3/24/2000 ****/

/**********
 * Copyright 2000 Regents of the University of California. All rights reserved.
 * File: b4ask.c of BSIM4.0.0.
 * Authors: Weidong Liu, Kanyu M. Cao, Xiaodong Jin, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "bsim4v0def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BSIM4v0ask(
CKTcircuit *ckt,
GENinstance *inst,
int which,
IFvalue *value,
IFvalue *select)
{
BSIM4v0instance *here = (BSIM4v0instance*)inst;

    NG_IGNORE(select);

    switch(which) 
    {   case BSIM4v0_L:
            value->rValue = here->BSIM4v0l;
            return(OK);
        case BSIM4v0_W:
            value->rValue = here->BSIM4v0w;
            return(OK);
        case BSIM4v0_NF:
            value->rValue = here->BSIM4v0nf;
            return(OK);
        case BSIM4v0_MIN:
            value->iValue = here->BSIM4v0min;
            return(OK);
        case BSIM4v0_AS:
            value->rValue = here->BSIM4v0sourceArea;
            return(OK);
        case BSIM4v0_AD:
            value->rValue = here->BSIM4v0drainArea;
            return(OK);
        case BSIM4v0_PS:
            value->rValue = here->BSIM4v0sourcePerimeter;
            return(OK);
        case BSIM4v0_PD:
            value->rValue = here->BSIM4v0drainPerimeter;
            return(OK);
        case BSIM4v0_NRS:
            value->rValue = here->BSIM4v0sourceSquares;
            return(OK);
        case BSIM4v0_NRD:
            value->rValue = here->BSIM4v0drainSquares;
            return(OK);
        case BSIM4v0_OFF:
            value->rValue = here->BSIM4v0off;
            return(OK);
        case BSIM4v0_RBSB:
            value->rValue = here->BSIM4v0rbsb;
            return(OK);
        case BSIM4v0_RBDB:
            value->rValue = here->BSIM4v0rbdb;
            return(OK);
        case BSIM4v0_RBPB:
            value->rValue = here->BSIM4v0rbpb;
            return(OK);
        case BSIM4v0_RBPS:
            value->rValue = here->BSIM4v0rbps;
            return(OK);
        case BSIM4v0_RBPD:
            value->rValue = here->BSIM4v0rbpd;
            return(OK);
        case BSIM4v0_TRNQSMOD:
            value->iValue = here->BSIM4v0trnqsMod;
            return(OK);
        case BSIM4v0_ACNQSMOD:
            value->iValue = here->BSIM4v0acnqsMod;
            return(OK);
        case BSIM4v0_RBODYMOD:
            value->iValue = here->BSIM4v0rbodyMod;
            return(OK);
        case BSIM4v0_RGATEMOD:
            value->iValue = here->BSIM4v0rgateMod;
            return(OK);
        case BSIM4v0_GEOMOD:
            value->iValue = here->BSIM4v0geoMod;
            return(OK);
        case BSIM4v0_RGEOMOD:
            value->iValue = here->BSIM4v0rgeoMod;
            return(OK);
        case BSIM4v0_IC_VDS:
            value->rValue = here->BSIM4v0icVDS;
            return(OK);
        case BSIM4v0_IC_VGS:
            value->rValue = here->BSIM4v0icVGS;
            return(OK);
        case BSIM4v0_IC_VBS:
            value->rValue = here->BSIM4v0icVBS;
            return(OK);
        case BSIM4v0_DNODE:
            value->iValue = here->BSIM4v0dNode;
            return(OK);
        case BSIM4v0_GNODEEXT:
            value->iValue = here->BSIM4v0gNodeExt;
            return(OK);
        case BSIM4v0_SNODE:
            value->iValue = here->BSIM4v0sNode;
            return(OK);
        case BSIM4v0_BNODE:
            value->iValue = here->BSIM4v0bNode;
            return(OK);
        case BSIM4v0_DNODEPRIME:
            value->iValue = here->BSIM4v0dNodePrime;
            return(OK);
        case BSIM4v0_GNODEPRIME:
            value->iValue = here->BSIM4v0gNodePrime;
            return(OK);
        case BSIM4v0_GNODEMID:
            value->iValue = here->BSIM4v0gNodeMid;
            return(OK);
        case BSIM4v0_SNODEPRIME:
            value->iValue = here->BSIM4v0sNodePrime;
            return(OK);
        case BSIM4v0_DBNODE:
            value->iValue = here->BSIM4v0dbNode;
            return(OK);
        case BSIM4v0_BNODEPRIME:
            value->iValue = here->BSIM4v0bNodePrime;
            return(OK);
        case BSIM4v0_SBNODE:
            value->iValue = here->BSIM4v0sbNode;
            return(OK);
        case BSIM4v0_SOURCECONDUCT:
            value->rValue = here->BSIM4v0sourceConductance;
            return(OK);
        case BSIM4v0_DRAINCONDUCT:
            value->rValue = here->BSIM4v0drainConductance;
            return(OK);
        case BSIM4v0_VBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v0vbd);
            return(OK);
        case BSIM4v0_VBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v0vbs);
            return(OK);
        case BSIM4v0_VGS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v0vgs);
            return(OK);
        case BSIM4v0_VDS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v0vds);
            return(OK);
        case BSIM4v0_CD:
            value->rValue = here->BSIM4v0cd; 
            return(OK);
        case BSIM4v0_CBS:
            value->rValue = here->BSIM4v0cbs; 
            return(OK);
        case BSIM4v0_CBD:
            value->rValue = here->BSIM4v0cbd; 
            return(OK);
        case BSIM4v0_GM:
            value->rValue = here->BSIM4v0gm; 
            return(OK);
        case BSIM4v0_GDS:
            value->rValue = here->BSIM4v0gds; 
            return(OK);
        case BSIM4v0_GMBS:
            value->rValue = here->BSIM4v0gmbs; 
            return(OK);
        case BSIM4v0_GBD:
            value->rValue = here->BSIM4v0gbd; 
            return(OK);
        case BSIM4v0_GBS:
            value->rValue = here->BSIM4v0gbs; 
            return(OK);
        case BSIM4v0_QB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v0qb); 
            return(OK);
        case BSIM4v0_CQB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v0cqb); 
            return(OK);
        case BSIM4v0_QG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v0qg); 
            return(OK);
        case BSIM4v0_CQG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v0cqg); 
            return(OK);
        case BSIM4v0_QD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v0qd); 
            return(OK);
        case BSIM4v0_CQD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v0cqd); 
            return(OK);
        case BSIM4v0_CGG:
            value->rValue = here->BSIM4v0cggb; 
            return(OK);
        case BSIM4v0_CGD:
            value->rValue = here->BSIM4v0cgdb;
            return(OK);
        case BSIM4v0_CGS:
            value->rValue = here->BSIM4v0cgsb;
            return(OK);
        case BSIM4v0_CDG:
            value->rValue = here->BSIM4v0cdgb; 
            return(OK);
        case BSIM4v0_CDD:
            value->rValue = here->BSIM4v0cddb; 
            return(OK);
        case BSIM4v0_CDS:
            value->rValue = here->BSIM4v0cdsb; 
            return(OK);
        case BSIM4v0_CBG:
            value->rValue = here->BSIM4v0cbgb;
            return(OK);
        case BSIM4v0_CBDB:
            value->rValue = here->BSIM4v0cbdb;
            return(OK);
        case BSIM4v0_CBSB:
            value->rValue = here->BSIM4v0cbsb;
            return(OK);
        case BSIM4v0_CAPBD:
            value->rValue = here->BSIM4v0capbd; 
            return(OK);
        case BSIM4v0_CAPBS:
            value->rValue = here->BSIM4v0capbs;
            return(OK);
        case BSIM4v0_VON:
            value->rValue = here->BSIM4v0von; 
            return(OK);
        case BSIM4v0_VDSAT:
            value->rValue = here->BSIM4v0vdsat; 
            return(OK);
        case BSIM4v0_QBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v0qbs); 
            return(OK);
        case BSIM4v0_QBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v0qbd); 
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

