/**** BSIM4.5.0 Released by Xuemei (Jane) Xi 07/27/2005 ****/

/**********
 * Copyright 2005 Regents of the University of California. All rights reserved.
 * File: b4ask.c of BSIM4.5.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 05/09/2003.
 * Modified by Xuemei Xi, Mohan Dunga, 07/29/2005.
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "bsim4v5def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BSIM4v5ask(
CKTcircuit *ckt,
GENinstance *inst,
int which,
IFvalue *value,
IFvalue *select)
{
BSIM4v5instance *here = (BSIM4v5instance*)inst;

    NG_IGNORE(select);

    switch(which) 
    {   case BSIM4v5_L:
            value->rValue = here->BSIM4v5l;
            return(OK);
        case BSIM4v5_W:
            value->rValue = here->BSIM4v5w;
            return(OK);
        case BSIM4v5_M:
            value->rValue = here->BSIM4v5m;
            return(OK);
        case BSIM4v5_NF:
            value->rValue = here->BSIM4v5nf;
            return(OK);
        case BSIM4v5_MIN:
            value->iValue = here->BSIM4v5min;
            return(OK);
        case BSIM4v5_AS:
            value->rValue = here->BSIM4v5sourceArea;
            return(OK);
        case BSIM4v5_AD:
            value->rValue = here->BSIM4v5drainArea;
            return(OK);
        case BSIM4v5_PS:
            value->rValue = here->BSIM4v5sourcePerimeter;
            return(OK);
        case BSIM4v5_PD:
            value->rValue = here->BSIM4v5drainPerimeter;
            return(OK);
        case BSIM4v5_NRS:
            value->rValue = here->BSIM4v5sourceSquares;
            return(OK);
        case BSIM4v5_NRD:
            value->rValue = here->BSIM4v5drainSquares;
            return(OK);
        case BSIM4v5_OFF:
            value->rValue = here->BSIM4v5off;
            return(OK);
        case BSIM4v5_SA:
            value->rValue = here->BSIM4v5sa ;
            return(OK);
        case BSIM4v5_SB:
            value->rValue = here->BSIM4v5sb ;
            return(OK);
        case BSIM4v5_SD:
            value->rValue = here->BSIM4v5sd ;
            return(OK);
	case BSIM4v5_SCA:
            value->rValue = here->BSIM4v5sca ;
            return(OK);
	case BSIM4v5_SCB:
            value->rValue = here->BSIM4v5scb ;
            return(OK);
	case BSIM4v5_SCC:
            value->rValue = here->BSIM4v5scc ;
            return(OK);
	case BSIM4v5_SC:
            value->rValue = here->BSIM4v5sc ;
            return(OK);

        case BSIM4v5_RBSB:
            value->rValue = here->BSIM4v5rbsb;
            return(OK);
        case BSIM4v5_RBDB:
            value->rValue = here->BSIM4v5rbdb;
            return(OK);
        case BSIM4v5_RBPB:
            value->rValue = here->BSIM4v5rbpb;
            return(OK);
        case BSIM4v5_RBPS:
            value->rValue = here->BSIM4v5rbps;
            return(OK);
        case BSIM4v5_RBPD:
            value->rValue = here->BSIM4v5rbpd;
            return(OK);
        case BSIM4v5_DELVTO:
            value->rValue = here->BSIM4v5delvto;
            return(OK);
        case BSIM4v5_MULU0:
            value->rValue = here->BSIM4v5mulu0;
            return(OK);
        case BSIM4v5_XGW:
            value->rValue = here->BSIM4v5xgw;
            return(OK);
        case BSIM4v5_NGCON:
            value->rValue = here->BSIM4v5ngcon;
            return(OK);
        case BSIM4v5_TRNQSMOD:
            value->iValue = here->BSIM4v5trnqsMod;
            return(OK);
        case BSIM4v5_ACNQSMOD:
            value->iValue = here->BSIM4v5acnqsMod;
            return(OK);
        case BSIM4v5_RBODYMOD:
            value->iValue = here->BSIM4v5rbodyMod;
            return(OK);
        case BSIM4v5_RGATEMOD:
            value->iValue = here->BSIM4v5rgateMod;
            return(OK);
        case BSIM4v5_GEOMOD:
            value->iValue = here->BSIM4v5geoMod;
            return(OK);
        case BSIM4v5_RGEOMOD:
            value->iValue = here->BSIM4v5rgeoMod;
            return(OK);
        case BSIM4v5_IC_VDS:
            value->rValue = here->BSIM4v5icVDS;
            return(OK);
        case BSIM4v5_IC_VGS:
            value->rValue = here->BSIM4v5icVGS;
            return(OK);
        case BSIM4v5_IC_VBS:
            value->rValue = here->BSIM4v5icVBS;
            return(OK);
        case BSIM4v5_DNODE:
            value->iValue = here->BSIM4v5dNode;
            return(OK);
        case BSIM4v5_GNODEEXT:
            value->iValue = here->BSIM4v5gNodeExt;
            return(OK);
        case BSIM4v5_SNODE:
            value->iValue = here->BSIM4v5sNode;
            return(OK);
        case BSIM4v5_BNODE:
            value->iValue = here->BSIM4v5bNode;
            return(OK);
        case BSIM4v5_DNODEPRIME:
            value->iValue = here->BSIM4v5dNodePrime;
            return(OK);
        case BSIM4v5_GNODEPRIME:
            value->iValue = here->BSIM4v5gNodePrime;
            return(OK);
        case BSIM4v5_GNODEMID:
            value->iValue = here->BSIM4v5gNodeMid;
            return(OK);
        case BSIM4v5_SNODEPRIME:
            value->iValue = here->BSIM4v5sNodePrime;
            return(OK);
        case BSIM4v5_DBNODE:
            value->iValue = here->BSIM4v5dbNode;
            return(OK);
        case BSIM4v5_BNODEPRIME:
            value->iValue = here->BSIM4v5bNodePrime;
            return(OK);
        case BSIM4v5_SBNODE:
            value->iValue = here->BSIM4v5sbNode;
            return(OK);
        case BSIM4v5_SOURCECONDUCT:
            value->rValue = here->BSIM4v5sourceConductance;
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_DRAINCONDUCT:
            value->rValue = here->BSIM4v5drainConductance;
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_VBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v5vbd);
            return(OK);
        case BSIM4v5_VBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v5vbs);
            return(OK);
        case BSIM4v5_VGS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v5vgs);
            return(OK);
        case BSIM4v5_VDS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v5vds);
            return(OK);
        case BSIM4v5_CD:
            value->rValue = here->BSIM4v5cd; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CBS:
            value->rValue = here->BSIM4v5cbs; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CBD:
            value->rValue = here->BSIM4v5cbd; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CSUB:
            value->rValue = here->BSIM4v5csub; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_QINV:
            value->rValue = here-> BSIM4v5qinv; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_IGIDL:
            value->rValue = here->BSIM4v5Igidl; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_IGISL:
            value->rValue = here->BSIM4v5Igisl; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_IGS:
            value->rValue = here->BSIM4v5Igs; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_IGD:
            value->rValue = here->BSIM4v5Igd; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_IGB:
            value->rValue = here->BSIM4v5Igb; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_IGCS:
            value->rValue = here->BSIM4v5Igcs; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_IGCD:
            value->rValue = here->BSIM4v5Igcd; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_GM:
            value->rValue = here->BSIM4v5gm; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_GDS:
            value->rValue = here->BSIM4v5gds; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_GMBS:
            value->rValue = here->BSIM4v5gmbs; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_GBD:
            value->rValue = here->BSIM4v5gbd; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_GBS:
            value->rValue = here->BSIM4v5gbs; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
/*        case BSIM4v5_QB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v5qb); 
            return(OK); */
        case BSIM4v5_CQB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v5cqb); 
            return(OK);
/*        case BSIM4v5_QG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v5qg); 
            return(OK); */
        case BSIM4v5_CQG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v5cqg); 
            return(OK);
/*        case BSIM4v5_QD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v5qd); 
            return(OK); */
        case BSIM4v5_CQD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v5cqd); 
            return(OK);
/*        case BSIM4v5_QS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v5qs); 
            return(OK); */
        case BSIM4v5_QB:
            value->rValue = here->BSIM4v5qbulk; 
            value->rValue *= here->BSIM4v5m;
            return(OK); 
        case BSIM4v5_QG:
            value->rValue = here->BSIM4v5qgate; 
            value->rValue *= here->BSIM4v5m;
            return(OK); 
        case BSIM4v5_QS:
            value->rValue = here->BSIM4v5qsrc; 
            value->rValue *= here->BSIM4v5m;
            return(OK); 
        case BSIM4v5_QD:
            value->rValue = here->BSIM4v5qdrn; 
            value->rValue *= here->BSIM4v5m;
            return(OK); 
        case BSIM4v5_QDEF:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v5qdef); 
            return(OK); 
        case BSIM4v5_GCRG:
            value->rValue = here->BSIM4v5gcrg;
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_GTAU:
            value->rValue = here->BSIM4v5gtau;
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CGGB:
            value->rValue = here->BSIM4v5cggb; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CGDB:
            value->rValue = here->BSIM4v5cgdb;
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CGSB:
            value->rValue = here->BSIM4v5cgsb;
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CDGB:
            value->rValue = here->BSIM4v5cdgb; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CDDB:
            value->rValue = here->BSIM4v5cddb; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CDSB:
            value->rValue = here->BSIM4v5cdsb; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CBGB:
            value->rValue = here->BSIM4v5cbgb;
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CBDB:
            value->rValue = here->BSIM4v5cbdb;
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CBSB:
            value->rValue = here->BSIM4v5cbsb;
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CSGB:
            value->rValue = here->BSIM4v5csgb;
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CSDB:
            value->rValue = here->BSIM4v5csdb;
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CSSB:
            value->rValue = here->BSIM4v5cssb;
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CGBB:
            value->rValue = here->BSIM4v5cgbb;
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CDBB:
            value->rValue = here->BSIM4v5cdbb;
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CSBB:
            value->rValue = here->BSIM4v5csbb;
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CBBB:
            value->rValue = here->BSIM4v5cbbb;
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CAPBD:
            value->rValue = here->BSIM4v5capbd; 
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_CAPBS:
            value->rValue = here->BSIM4v5capbs;
            value->rValue *= here->BSIM4v5m;
            return(OK);
        case BSIM4v5_VON:
            value->rValue = here->BSIM4v5von; 
            return(OK);
        case BSIM4v5_VDSAT:
            value->rValue = here->BSIM4v5vdsat; 
            return(OK);
        case BSIM4v5_QBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v5qbs); 
            return(OK);
        case BSIM4v5_QBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v5qbd); 
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

