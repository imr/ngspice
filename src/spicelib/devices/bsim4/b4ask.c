/**** BSIM4.1.0, Released by Weidong Liu 10/11/2000 ****/

/**********
 * Copyright 2000 Regents of the University of California. All rights reserved.
 * File: b4ask.c of BSIM4.1.0.
 * Authors: Weidong Liu, Kanyu M. Cao, Xiaodong Jin, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 *
 * Modified by Weidong Liu, 10/11/2000.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "bsim4def.h"
#include "sperror.h"

int
BSIM4ask(ckt,inst,which,value,select)
CKTcircuit *ckt;
GENinstance *inst;
int which;
IFvalue *value;
IFvalue *select;
{
BSIM4instance *here = (BSIM4instance*)inst;

    switch(which) 
    {   case BSIM4_L:
            value->rValue = here->BSIM4l;
            return(OK);
        case BSIM4_W:
            value->rValue = here->BSIM4w;
            return(OK);
        case BSIM4_NF:
            value->rValue = here->BSIM4nf;
            return(OK);
        case BSIM4_MIN:
            value->iValue = here->BSIM4min;
            return(OK);
        case BSIM4_AS:
            value->rValue = here->BSIM4sourceArea;
            return(OK);
        case BSIM4_AD:
            value->rValue = here->BSIM4drainArea;
            return(OK);
        case BSIM4_PS:
            value->rValue = here->BSIM4sourcePerimeter;
            return(OK);
        case BSIM4_PD:
            value->rValue = here->BSIM4drainPerimeter;
            return(OK);
        case BSIM4_NRS:
            value->rValue = here->BSIM4sourceSquares;
            return(OK);
        case BSIM4_NRD:
            value->rValue = here->BSIM4drainSquares;
            return(OK);
        case BSIM4_OFF:
            value->rValue = here->BSIM4off;
            return(OK);
        case BSIM4_RBSB:
            value->rValue = here->BSIM4rbsb;
            return(OK);
        case BSIM4_RBDB:
            value->rValue = here->BSIM4rbdb;
            return(OK);
        case BSIM4_RBPB:
            value->rValue = here->BSIM4rbpb;
            return(OK);
        case BSIM4_RBPS:
            value->rValue = here->BSIM4rbps;
            return(OK);
        case BSIM4_RBPD:
            value->rValue = here->BSIM4rbpd;
            return(OK);
        case BSIM4_TRNQSMOD:
            value->iValue = here->BSIM4trnqsMod;
            return(OK);
        case BSIM4_ACNQSMOD:
            value->iValue = here->BSIM4acnqsMod;
            return(OK);
        case BSIM4_RBODYMOD:
            value->iValue = here->BSIM4rbodyMod;
            return(OK);
        case BSIM4_RGATEMOD:
            value->iValue = here->BSIM4rgateMod;
            return(OK);
        case BSIM4_GEOMOD:
            value->iValue = here->BSIM4geoMod;
            return(OK);
        case BSIM4_RGEOMOD:
            value->iValue = here->BSIM4rgeoMod;
            return(OK);
        case BSIM4_IC_VDS:
            value->rValue = here->BSIM4icVDS;
            return(OK);
        case BSIM4_IC_VGS:
            value->rValue = here->BSIM4icVGS;
            return(OK);
        case BSIM4_IC_VBS:
            value->rValue = here->BSIM4icVBS;
            return(OK);
        case BSIM4_DNODE:
            value->iValue = here->BSIM4dNode;
            return(OK);
        case BSIM4_GNODEEXT:
            value->iValue = here->BSIM4gNodeExt;
            return(OK);
        case BSIM4_SNODE:
            value->iValue = here->BSIM4sNode;
            return(OK);
        case BSIM4_BNODE:
            value->iValue = here->BSIM4bNode;
            return(OK);
        case BSIM4_DNODEPRIME:
            value->iValue = here->BSIM4dNodePrime;
            return(OK);
        case BSIM4_GNODEPRIME:
            value->iValue = here->BSIM4gNodePrime;
            return(OK);
        case BSIM4_GNODEMID:
            value->iValue = here->BSIM4gNodeMid;
            return(OK);
        case BSIM4_SNODEPRIME:
            value->iValue = here->BSIM4sNodePrime;
            return(OK);
        case BSIM4_DBNODE:
            value->iValue = here->BSIM4dbNode;
            return(OK);
        case BSIM4_BNODEPRIME:
            value->iValue = here->BSIM4bNodePrime;
            return(OK);
        case BSIM4_SBNODE:
            value->iValue = here->BSIM4sbNode;
            return(OK);
        case BSIM4_SOURCECONDUCT:
            value->rValue = here->BSIM4sourceConductance;
            return(OK);
        case BSIM4_DRAINCONDUCT:
            value->rValue = here->BSIM4drainConductance;
            return(OK);
        case BSIM4_VBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4vbd);
            return(OK);
        case BSIM4_VBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4vbs);
            return(OK);
        case BSIM4_VGS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4vgs);
            return(OK);
        case BSIM4_VDS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4vds);
            return(OK);
        case BSIM4_CD:
            value->rValue = here->BSIM4cd; 
            return(OK);
        case BSIM4_CBS:
            value->rValue = here->BSIM4cbs; 
            return(OK);
        case BSIM4_CBD:
            value->rValue = here->BSIM4cbd; 
            return(OK);
        case BSIM4_GM:
            value->rValue = here->BSIM4gm; 
            return(OK);
        case BSIM4_GDS:
            value->rValue = here->BSIM4gds; 
            return(OK);
        case BSIM4_GMBS:
            value->rValue = here->BSIM4gmbs; 
            return(OK);
        case BSIM4_GBD:
            value->rValue = here->BSIM4gbd; 
            return(OK);
        case BSIM4_GBS:
            value->rValue = here->BSIM4gbs; 
            return(OK);
        case BSIM4_QB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4qb); 
            return(OK);
        case BSIM4_CQB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4cqb); 
            return(OK);
        case BSIM4_QG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4qg); 
            return(OK);
        case BSIM4_CQG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4cqg); 
            return(OK);
        case BSIM4_QD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4qd); 
            return(OK);
        case BSIM4_CQD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4cqd); 
            return(OK);
        case BSIM4_CGG:
            value->rValue = here->BSIM4cggb; 
            return(OK);
        case BSIM4_CGD:
            value->rValue = here->BSIM4cgdb;
            return(OK);
        case BSIM4_CGS:
            value->rValue = here->BSIM4cgsb;
            return(OK);
        case BSIM4_CDG:
            value->rValue = here->BSIM4cdgb; 
            return(OK);
        case BSIM4_CDD:
            value->rValue = here->BSIM4cddb; 
            return(OK);
        case BSIM4_CDS:
            value->rValue = here->BSIM4cdsb; 
            return(OK);
        case BSIM4_CBG:
            value->rValue = here->BSIM4cbgb;
            return(OK);
        case BSIM4_CBDB:
            value->rValue = here->BSIM4cbdb;
            return(OK);
        case BSIM4_CBSB:
            value->rValue = here->BSIM4cbsb;
            return(OK);
        case BSIM4_CAPBD:
            value->rValue = here->BSIM4capbd; 
            return(OK);
        case BSIM4_CAPBS:
            value->rValue = here->BSIM4capbs;
            return(OK);
        case BSIM4_VON:
            value->rValue = here->BSIM4von; 
            return(OK);
        case BSIM4_VDSAT:
            value->rValue = here->BSIM4vdsat; 
            return(OK);
        case BSIM4_QBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4qbs); 
            return(OK);
        case BSIM4_QBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4qbd); 
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

