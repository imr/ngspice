/**** BSIM4.3.0 Released by Xuemei (Jane) Xi  05/09/2003 ****/

/**********
 * Copyright 2003 Regents of the University of California. All rights reserved.
 * File: b4v3ask.c of BSIM4.3.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 05/09/2003.
 **********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "bsim4v3def.h"
#include "sperror.h"

int
BSIM4v3ask(
CKTcircuit *ckt,
GENinstance *inst,
int which,
IFvalue *value,
IFvalue *select)
{
BSIM4v3instance *here = (BSIM4v3instance*)inst;

    switch(which) 
    {   case BSIM4v3_L:
            value->rValue = here->BSIM4v3l;
            return(OK);
        case BSIM4v3_W:
            value->rValue = here->BSIM4v3w;
            return(OK);
        case BSIM4v3_M:
            value->rValue = here->BSIM4v3m;
            return(OK);
        case BSIM4v3_NF:
            value->rValue = here->BSIM4v3nf;
            return(OK);
        case BSIM4v3_MIN:
            value->iValue = here->BSIM4v3min;
            return(OK);
        case BSIM4v3_AS:
            value->rValue = here->BSIM4v3sourceArea;
            return(OK);
        case BSIM4v3_AD:
            value->rValue = here->BSIM4v3drainArea;
            return(OK);
        case BSIM4v3_PS:
            value->rValue = here->BSIM4v3sourcePerimeter;
            return(OK);
        case BSIM4v3_PD:
            value->rValue = here->BSIM4v3drainPerimeter;
            return(OK);
        case BSIM4v3_NRS:
            value->rValue = here->BSIM4v3sourceSquares;
            return(OK);
        case BSIM4v3_NRD:
            value->rValue = here->BSIM4v3drainSquares;
            return(OK);
        case BSIM4v3_OFF:
            value->rValue = here->BSIM4v3off;
            return(OK);
        case BSIM4v3_SA:
            value->rValue = here->BSIM4v3sa ;
            return(OK);
        case BSIM4v3_SB:
            value->rValue = here->BSIM4v3sb ;
            return(OK);
        case BSIM4v3_SD:
            value->rValue = here->BSIM4v3sd ;
            return(OK);
        case BSIM4v3_RBSB:
            value->rValue = here->BSIM4v3rbsb;
            return(OK);
        case BSIM4v3_RBDB:
            value->rValue = here->BSIM4v3rbdb;
            return(OK);
        case BSIM4v3_RBPB:
            value->rValue = here->BSIM4v3rbpb;
            return(OK);
        case BSIM4v3_RBPS:
            value->rValue = here->BSIM4v3rbps;
            return(OK);
        case BSIM4v3_RBPD:
            value->rValue = here->BSIM4v3rbpd;
            return(OK);
        case BSIM4v3_TRNQSMOD:
            value->iValue = here->BSIM4v3trnqsMod;
            return(OK);
        case BSIM4v3_ACNQSMOD:
            value->iValue = here->BSIM4v3acnqsMod;
            return(OK);
        case BSIM4v3_RBODYMOD:
            value->iValue = here->BSIM4v3rbodyMod;
            return(OK);
        case BSIM4v3_RGATEMOD:
            value->iValue = here->BSIM4v3rgateMod;
            return(OK);
        case BSIM4v3_GEOMOD:
            value->iValue = here->BSIM4v3geoMod;
            return(OK);
        case BSIM4v3_RGEOMOD:
            value->iValue = here->BSIM4v3rgeoMod;
            return(OK);
        case BSIM4v3_IC_VDS:
            value->rValue = here->BSIM4v3icVDS;
            return(OK);
        case BSIM4v3_IC_VGS:
            value->rValue = here->BSIM4v3icVGS;
            return(OK);
        case BSIM4v3_IC_VBS:
            value->rValue = here->BSIM4v3icVBS;
            return(OK);
        case BSIM4v3_DNODE:
            value->iValue = here->BSIM4v3dNode;
            return(OK);
        case BSIM4v3_GNODEEXT:
            value->iValue = here->BSIM4v3gNodeExt;
            return(OK);
        case BSIM4v3_SNODE:
            value->iValue = here->BSIM4v3sNode;
            return(OK);
        case BSIM4v3_BNODE:
            value->iValue = here->BSIM4v3bNode;
            return(OK);
        case BSIM4v3_DNODEPRIME:
            value->iValue = here->BSIM4v3dNodePrime;
            return(OK);
        case BSIM4v3_GNODEPRIME:
            value->iValue = here->BSIM4v3gNodePrime;
            return(OK);
        case BSIM4v3_GNODEMID:
            value->iValue = here->BSIM4v3gNodeMid;
            return(OK);
        case BSIM4v3_SNODEPRIME:
            value->iValue = here->BSIM4v3sNodePrime;
            return(OK);
        case BSIM4v3_DBNODE:
            value->iValue = here->BSIM4v3dbNode;
            return(OK);
        case BSIM4v3_BNODEPRIME:
            value->iValue = here->BSIM4v3bNodePrime;
            return(OK);
        case BSIM4v3_SBNODE:
            value->iValue = here->BSIM4v3sbNode;
            return(OK);
        case BSIM4v3_SOURCECONDUCT:
            value->rValue = here->BSIM4v3sourceConductance;
            return(OK);
        case BSIM4v3_DRAINCONDUCT:
            value->rValue = here->BSIM4v3drainConductance;
            return(OK);
        case BSIM4v3_VBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v3vbd);
            return(OK);
        case BSIM4v3_VBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v3vbs);
            return(OK);
        case BSIM4v3_VGS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v3vgs);
            return(OK);
        case BSIM4v3_VDS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v3vds);
            return(OK);
        case BSIM4v3_CD:
            value->rValue = here->BSIM4v3cd; 
            return(OK);
        case BSIM4v3_CBS:
            value->rValue = here->BSIM4v3cbs; 
            return(OK);
        case BSIM4v3_CBD:
            value->rValue = here->BSIM4v3cbd; 
            return(OK);
        case BSIM4v3_CSUB:
            value->rValue = here->BSIM4v3csub; 
            return(OK);
        case BSIM4v3_QINV:
            value->rValue = here-> BSIM4v3qinv; 
            return(OK);
        case BSIM4v3_IGIDL:
            value->rValue = here->BSIM4v3Igidl; 
            return(OK);
        case BSIM4v3_IGISL:
            value->rValue = here->BSIM4v3Igisl; 
            return(OK);
        case BSIM4v3_IGS:
            value->rValue = here->BSIM4v3Igs; 
            return(OK);
        case BSIM4v3_IGD:
            value->rValue = here->BSIM4v3Igd; 
            return(OK);
        case BSIM4v3_IGB:
            value->rValue = here->BSIM4v3Igb; 
            return(OK);
        case BSIM4v3_IGCS:
            value->rValue = here->BSIM4v3Igcs; 
            return(OK);
        case BSIM4v3_IGCD:
            value->rValue = here->BSIM4v3Igcd; 
            return(OK);
        case BSIM4v3_GM:
            value->rValue = here->BSIM4v3gm; 
            return(OK);
        case BSIM4v3_GDS:
            value->rValue = here->BSIM4v3gds; 
            return(OK);
        case BSIM4v3_GMBS:
            value->rValue = here->BSIM4v3gmbs; 
            return(OK);
        case BSIM4v3_GBD:
            value->rValue = here->BSIM4v3gbd; 
            return(OK);
        case BSIM4v3_GBS:
            value->rValue = here->BSIM4v3gbs; 
            return(OK);
        case BSIM4v3_QB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v3qb); 
            return(OK); 
        case BSIM4v3_CQB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v3cqb); 
            return(OK);
        case BSIM4v3_QG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v3qg); 
            return(OK);
        case BSIM4v3_CQG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v3cqg); 
            return(OK);
        case BSIM4v3_QD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v3qd); 
            return(OK); 
        case BSIM4v3_CQD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v3cqd); 
            return(OK);
        case BSIM4v3_QS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v3qs); 
            return(OK); 
        case BSIM4v3_CGGB:
            value->rValue = here->BSIM4v3cggb; 
            return(OK);
        case BSIM4v3_CGDB:
            value->rValue = here->BSIM4v3cgdb;
            return(OK);
        case BSIM4v3_CGSB:
            value->rValue = here->BSIM4v3cgsb;
            return(OK);
        case BSIM4v3_CDGB:
            value->rValue = here->BSIM4v3cdgb; 
            return(OK);
        case BSIM4v3_CDDB:
            value->rValue = here->BSIM4v3cddb; 
            return(OK);
        case BSIM4v3_CDSB:
            value->rValue = here->BSIM4v3cdsb; 
            return(OK);
        case BSIM4v3_CBGB:
            value->rValue = here->BSIM4v3cbgb;
            return(OK);
        case BSIM4v3_CBDB:
            value->rValue = here->BSIM4v3cbdb;
            return(OK);
        case BSIM4v3_CBSB:
            value->rValue = here->BSIM4v3cbsb;
            return(OK);
        case BSIM4v3_CSGB:
            value->rValue = here->BSIM4v3csgb;
            return(OK);
        case BSIM4v3_CSDB:
            value->rValue = here->BSIM4v3csdb;
            return(OK);
        case BSIM4v3_CSSB:
            value->rValue = here->BSIM4v3cssb;
            return(OK);
        case BSIM4v3_CGBB:
            value->rValue = here->BSIM4v3cgbb;
            return(OK);
        case BSIM4v3_CDBB:
            value->rValue = here->BSIM4v3cdbb;
            return(OK);
        case BSIM4v3_CSBB:
            value->rValue = here->BSIM4v3csbb;
            return(OK);
        case BSIM4v3_CBBB:
            value->rValue = here->BSIM4v3cbbb;
            return(OK);
        case BSIM4v3_CAPBD:
            value->rValue = here->BSIM4v3capbd; 
            return(OK);
        case BSIM4v3_CAPBS:
            value->rValue = here->BSIM4v3capbs;
            return(OK);
        case BSIM4v3_VON:
            value->rValue = here->BSIM4v3von; 
            return(OK);
        case BSIM4v3_VDSAT:
            value->rValue = here->BSIM4v3vdsat; 
            return(OK);
        case BSIM4v3_QBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v3qbs); 
            return(OK);
        case BSIM4v3_QBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v3qbd); 
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

