/**** BSIM4.2.1, Released by Xuemei Xi 10/05/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b4ask.c of BSIM4.2.1.
 * Author: 2000 Weidong Liu
 * Authors: Xuemei Xi, Kanyu M. Cao, Hui Wan, Mansun Chan, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 *
 * Modified by Xuemei Xi, 10/05/2001.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "bsim4v2def.h"
#include "sperror.h"


int
BSIM4v2ask(
CKTcircuit *ckt,
GENinstance *inst,
int which,
IFvalue *value,
IFvalue *select)
{
BSIM4v2instance *here = (BSIM4v2instance*)inst;

    NG_IGNORE(select);

    switch(which) 
    {   case BSIM4v2_L:
            value->rValue = here->BSIM4v2l;
            return(OK);
        case BSIM4v2_W:
            value->rValue = here->BSIM4v2w;
            return(OK);
        case BSIM4v2_M:
            value->rValue = here->BSIM4v2m;
            return(OK);
        case BSIM4v2_NF:
            value->rValue = here->BSIM4v2nf;
            return(OK);
        case BSIM4v2_MIN:
            value->iValue = here->BSIM4v2min;
            return(OK);
        case BSIM4v2_AS:
            value->rValue = here->BSIM4v2sourceArea;
            return(OK);
        case BSIM4v2_AD:
            value->rValue = here->BSIM4v2drainArea;
            return(OK);
        case BSIM4v2_PS:
            value->rValue = here->BSIM4v2sourcePerimeter;
            return(OK);
        case BSIM4v2_PD:
            value->rValue = here->BSIM4v2drainPerimeter;
            return(OK);
        case BSIM4v2_NRS:
            value->rValue = here->BSIM4v2sourceSquares;
            return(OK);
        case BSIM4v2_NRD:
            value->rValue = here->BSIM4v2drainSquares;
            return(OK);
        case BSIM4v2_OFF:
            value->rValue = here->BSIM4v2off;
            return(OK);
        case BSIM4v2_RBSB:
            value->rValue = here->BSIM4v2rbsb;
            return(OK);
        case BSIM4v2_RBDB:
            value->rValue = here->BSIM4v2rbdb;
            return(OK);
        case BSIM4v2_RBPB:
            value->rValue = here->BSIM4v2rbpb;
            return(OK);
        case BSIM4v2_RBPS:
            value->rValue = here->BSIM4v2rbps;
            return(OK);
        case BSIM4v2_RBPD:
            value->rValue = here->BSIM4v2rbpd;
            return(OK);
        case BSIM4v2_TRNQSMOD:
            value->iValue = here->BSIM4v2trnqsMod;
            return(OK);
        case BSIM4v2_ACNQSMOD:
            value->iValue = here->BSIM4v2acnqsMod;
            return(OK);
        case BSIM4v2_RBODYMOD:
            value->iValue = here->BSIM4v2rbodyMod;
            return(OK);
        case BSIM4v2_RGATEMOD:
            value->iValue = here->BSIM4v2rgateMod;
            return(OK);
        case BSIM4v2_GEOMOD:
            value->iValue = here->BSIM4v2geoMod;
            return(OK);
        case BSIM4v2_RGEOMOD:
            value->iValue = here->BSIM4v2rgeoMod;
            return(OK);
        case BSIM4v2_IC_VDS:
            value->rValue = here->BSIM4v2icVDS;
            return(OK);
        case BSIM4v2_IC_VGS:
            value->rValue = here->BSIM4v2icVGS;
            return(OK);
        case BSIM4v2_IC_VBS:
            value->rValue = here->BSIM4v2icVBS;
            return(OK);
        case BSIM4v2_DNODE:
            value->iValue = here->BSIM4v2dNode;
            return(OK);
        case BSIM4v2_GNODEEXT:
            value->iValue = here->BSIM4v2gNodeExt;
            return(OK);
        case BSIM4v2_SNODE:
            value->iValue = here->BSIM4v2sNode;
            return(OK);
        case BSIM4v2_BNODE:
            value->iValue = here->BSIM4v2bNode;
            return(OK);
        case BSIM4v2_DNODEPRIME:
            value->iValue = here->BSIM4v2dNodePrime;
            return(OK);
        case BSIM4v2_GNODEPRIME:
            value->iValue = here->BSIM4v2gNodePrime;
            return(OK);
        case BSIM4v2_GNODEMID:
            value->iValue = here->BSIM4v2gNodeMid;
            return(OK);
        case BSIM4v2_SNODEPRIME:
            value->iValue = here->BSIM4v2sNodePrime;
            return(OK);
        case BSIM4v2_DBNODE:
            value->iValue = here->BSIM4v2dbNode;
            return(OK);
        case BSIM4v2_BNODEPRIME:
            value->iValue = here->BSIM4v2bNodePrime;
            return(OK);
        case BSIM4v2_SBNODE:
            value->iValue = here->BSIM4v2sbNode;
            return(OK);
        case BSIM4v2_SOURCECONDUCT:
            value->rValue = here->BSIM4v2sourceConductance;
            return(OK);
        case BSIM4v2_DRAINCONDUCT:
            value->rValue = here->BSIM4v2drainConductance;
            return(OK);
        case BSIM4v2_VBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v2vbd);
            return(OK);
        case BSIM4v2_VBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v2vbs);
            return(OK);
        case BSIM4v2_VGS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v2vgs);
            return(OK);
        case BSIM4v2_VDS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v2vds);
            return(OK);
        case BSIM4v2_CD:
            value->rValue = here->BSIM4v2cd; 
            return(OK);
        case BSIM4v2_CBS:
            value->rValue = here->BSIM4v2cbs; 
            return(OK);
        case BSIM4v2_CBD:
            value->rValue = here->BSIM4v2cbd; 
            return(OK);
        case BSIM4v2_CSUB:
            value->rValue = here->BSIM4v2csub; 
            return(OK);
        case BSIM4v2_QINV:
            value->rValue = here-> BSIM4v2qinv; 
            return(OK);
        case BSIM4v2_IGIDL:
            value->rValue = here->BSIM4v2Igidl; 
            return(OK);
        case BSIM4v2_IGISL:
            value->rValue = here->BSIM4v2Igisl; 
            return(OK);
        case BSIM4v2_IGS:
            value->rValue = here->BSIM4v2Igs; 
            return(OK);
        case BSIM4v2_IGD:
            value->rValue = here->BSIM4v2Igd; 
            return(OK);
        case BSIM4v2_IGB:
            value->rValue = here->BSIM4v2Igb; 
            return(OK);
        case BSIM4v2_IGCS:
            value->rValue = here->BSIM4v2Igcs; 
            return(OK);
        case BSIM4v2_IGCD:
            value->rValue = here->BSIM4v2Igcd; 
            return(OK);
        case BSIM4v2_GM:
            value->rValue = here->BSIM4v2gm; 
            return(OK);
        case BSIM4v2_GDS:
            value->rValue = here->BSIM4v2gds; 
            return(OK);
        case BSIM4v2_GMBS:
            value->rValue = here->BSIM4v2gmbs; 
            return(OK);
        case BSIM4v2_GBD:
            value->rValue = here->BSIM4v2gbd; 
            return(OK);
        case BSIM4v2_GBS:
            value->rValue = here->BSIM4v2gbs; 
            return(OK);
        case BSIM4v2_QB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v2qb); 
            return(OK); 
        case BSIM4v2_CQB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v2cqb); 
            return(OK);
        case BSIM4v2_QG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v2qg); 
            return(OK);
        case BSIM4v2_CQG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v2cqg); 
            return(OK);
        case BSIM4v2_QD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v2qd); 
            return(OK); 
        case BSIM4v2_CQD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v2cqd); 
            return(OK);
        case BSIM4v2_QS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v2qs); 
            return(OK); 
        case BSIM4v2_CGGB:
            value->rValue = here->BSIM4v2cggb; 
            return(OK);
        case BSIM4v2_CGDB:
            value->rValue = here->BSIM4v2cgdb;
            return(OK);
        case BSIM4v2_CGSB:
            value->rValue = here->BSIM4v2cgsb;
            return(OK);
        case BSIM4v2_CDGB:
            value->rValue = here->BSIM4v2cdgb; 
            return(OK);
        case BSIM4v2_CDDB:
            value->rValue = here->BSIM4v2cddb; 
            return(OK);
        case BSIM4v2_CDSB:
            value->rValue = here->BSIM4v2cdsb; 
            return(OK);
        case BSIM4v2_CBGB:
            value->rValue = here->BSIM4v2cbgb;
            return(OK);
        case BSIM4v2_CBDB:
            value->rValue = here->BSIM4v2cbdb;
            return(OK);
        case BSIM4v2_CBSB:
            value->rValue = here->BSIM4v2cbsb;
            return(OK);
        case BSIM4v2_CSGB:
            value->rValue = here->BSIM4v2csgb;
            return(OK);
        case BSIM4v2_CSDB:
            value->rValue = here->BSIM4v2csdb;
            return(OK);
        case BSIM4v2_CSSB:
            value->rValue = here->BSIM4v2cssb;
            return(OK);
        case BSIM4v2_CGBB:
            value->rValue = here->BSIM4v2cgbb;
            return(OK);
        case BSIM4v2_CDBB:
            value->rValue = here->BSIM4v2cdbb;
            return(OK);
        case BSIM4v2_CSBB:
            value->rValue = here->BSIM4v2csbb;
            return(OK);
        case BSIM4v2_CBBB:
            value->rValue = here->BSIM4v2cbbb;
            return(OK);
        case BSIM4v2_CAPBD:
            value->rValue = here->BSIM4v2capbd; 
            return(OK);
        case BSIM4v2_CAPBS:
            value->rValue = here->BSIM4v2capbs;
            return(OK);
        case BSIM4v2_VON:
            value->rValue = here->BSIM4v2von; 
            return(OK);
        case BSIM4v2_VDSAT:
            value->rValue = here->BSIM4v2vdsat; 
            return(OK);
        case BSIM4v2_QBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v2qbs); 
            return(OK);
        case BSIM4v2_QBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v2qbd); 
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

