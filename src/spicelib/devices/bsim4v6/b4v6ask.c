/**** BSIM4.6.2 Released by Wenwei Yang 07/31/2008 ****/

/**********
 * Copyright 2006 Regents of the University of California. All rights reserved.
 * File: b4ask.c of BSIM4.6.2.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Mohan Dunga, Ali Niknejad, Chenming Hu.
 * Authors: 2006- Mohan Dunga, Ali Niknejad, Chenming Hu
 * Authors: 2007- Mohan Dunga, Wenwei Yang, Ali Niknejad, Chenming Hu
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
#include "bsim4v6def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BSIM4v6ask(
CKTcircuit *ckt,
GENinstance *inst,
int which,
IFvalue *value,
IFvalue *select)
{
BSIM4v6instance *here = (BSIM4v6instance*)inst;

    NG_IGNORE(select);

    switch(which) 
    {   case BSIM4v6_L:
            value->rValue = here->BSIM4v6l;
            return(OK);
        case BSIM4v6_W:
            value->rValue = here->BSIM4v6w;
            return(OK);
        case BSIM4v6_M:
            value->rValue = here->BSIM4v6m;
            return(OK);
        case BSIM4v6_NF:
            value->rValue = here->BSIM4v6nf;
            return(OK);
        case BSIM4v6_MIN:
            value->iValue = here->BSIM4v6min;
            return(OK);
        case BSIM4v6_AS:
            value->rValue = here->BSIM4v6sourceArea;
            return(OK);
        case BSIM4v6_AD:
            value->rValue = here->BSIM4v6drainArea;
            return(OK);
        case BSIM4v6_PS:
            value->rValue = here->BSIM4v6sourcePerimeter;
            return(OK);
        case BSIM4v6_PD:
            value->rValue = here->BSIM4v6drainPerimeter;
            return(OK);
        case BSIM4v6_NRS:
            value->rValue = here->BSIM4v6sourceSquares;
            return(OK);
        case BSIM4v6_NRD:
            value->rValue = here->BSIM4v6drainSquares;
            return(OK);
        case BSIM4v6_OFF:
            value->rValue = here->BSIM4v6off;
            return(OK);
        case BSIM4v6_SA:
            value->rValue = here->BSIM4v6sa ;
            return(OK);
        case BSIM4v6_SB:
            value->rValue = here->BSIM4v6sb ;
            return(OK);
        case BSIM4v6_SD:
            value->rValue = here->BSIM4v6sd ;
            return(OK);
        case BSIM4v6_SCA:
            value->rValue = here->BSIM4v6sca ;
            return(OK);
        case BSIM4v6_SCB:
            value->rValue = here->BSIM4v6scb ;
            return(OK);
        case BSIM4v6_SCC:
            value->rValue = here->BSIM4v6scc ;
            return(OK);
        case BSIM4v6_SC:
            value->rValue = here->BSIM4v6sc ;
            return(OK);

        case BSIM4v6_RBSB:
            value->rValue = here->BSIM4v6rbsb;
            return(OK);
        case BSIM4v6_RBDB:
            value->rValue = here->BSIM4v6rbdb;
            return(OK);
        case BSIM4v6_RBPB:
            value->rValue = here->BSIM4v6rbpb;
            return(OK);
        case BSIM4v6_RBPS:
            value->rValue = here->BSIM4v6rbps;
            return(OK);
        case BSIM4v6_RBPD:
            value->rValue = here->BSIM4v6rbpd;
            return(OK);
        case BSIM4v6_DELVTO:
            value->rValue = here->BSIM4v6delvto;
            return(OK);
        case BSIM4v6_MULU0:
            value->rValue = here->BSIM4v6mulu0;
            return(OK);
        case BSIM4v6_XGW:
            value->rValue = here->BSIM4v6xgw;
            return(OK);
        case BSIM4v6_NGCON:
            value->rValue = here->BSIM4v6ngcon;
            return(OK);
        case BSIM4v6_TRNQSMOD:
            value->iValue = here->BSIM4v6trnqsMod;
            return(OK);
        case BSIM4v6_ACNQSMOD:
            value->iValue = here->BSIM4v6acnqsMod;
            return(OK);
        case BSIM4v6_RBODYMOD:
            value->iValue = here->BSIM4v6rbodyMod;
            return(OK);
        case BSIM4v6_RGATEMOD:
            value->iValue = here->BSIM4v6rgateMod;
            return(OK);
        case BSIM4v6_GEOMOD:
            value->iValue = here->BSIM4v6geoMod;
            return(OK);
        case BSIM4v6_RGEOMOD:
            value->iValue = here->BSIM4v6rgeoMod;
            return(OK);
        case BSIM4v6_IC_VDS:
            value->rValue = here->BSIM4v6icVDS;
            return(OK);
        case BSIM4v6_IC_VGS:
            value->rValue = here->BSIM4v6icVGS;
            return(OK);
        case BSIM4v6_IC_VBS:
            value->rValue = here->BSIM4v6icVBS;
            return(OK);
        case BSIM4v6_DNODE:
            value->iValue = here->BSIM4v6dNode;
            return(OK);
        case BSIM4v6_GNODEEXT:
            value->iValue = here->BSIM4v6gNodeExt;
            return(OK);
        case BSIM4v6_SNODE:
            value->iValue = here->BSIM4v6sNode;
            return(OK);
        case BSIM4v6_BNODE:
            value->iValue = here->BSIM4v6bNode;
            return(OK);
        case BSIM4v6_DNODEPRIME:
            value->iValue = here->BSIM4v6dNodePrime;
            return(OK);
        case BSIM4v6_GNODEPRIME:
            value->iValue = here->BSIM4v6gNodePrime;
            return(OK);
        case BSIM4v6_GNODEMID:
            value->iValue = here->BSIM4v6gNodeMid;
            return(OK);
        case BSIM4v6_SNODEPRIME:
            value->iValue = here->BSIM4v6sNodePrime;
            return(OK);
        case BSIM4v6_DBNODE:
            value->iValue = here->BSIM4v6dbNode;
            return(OK);
        case BSIM4v6_BNODEPRIME:
            value->iValue = here->BSIM4v6bNodePrime;
            return(OK);
        case BSIM4v6_SBNODE:
            value->iValue = here->BSIM4v6sbNode;
            return(OK);
        case BSIM4v6_SOURCECONDUCT:
            value->rValue = here->BSIM4v6sourceConductance;
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_DRAINCONDUCT:
            value->rValue = here->BSIM4v6drainConductance;
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_VBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v6vbd);
            return(OK);
        case BSIM4v6_VBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v6vbs);
            return(OK);
        case BSIM4v6_VGS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v6vgs);
            return(OK);
        case BSIM4v6_VDS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v6vds);
            return(OK);
        case BSIM4v6_CD:
            value->rValue = here->BSIM4v6cd; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_CBS:
            value->rValue = here->BSIM4v6cbs; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_CBD:
            value->rValue = here->BSIM4v6cbd; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_CSUB:
            value->rValue = here->BSIM4v6csub; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_QINV:
            value->rValue = here-> BSIM4v6qinv; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_IGIDL:
            value->rValue = here->BSIM4v6Igidl; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_IGISL:
            value->rValue = here->BSIM4v6Igisl; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_IGS:
            value->rValue = here->BSIM4v6Igs; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_IGD:
            value->rValue = here->BSIM4v6Igd; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_IGB:
            value->rValue = here->BSIM4v6Igb; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_IGCS:
            value->rValue = here->BSIM4v6Igcs; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_IGCD:
            value->rValue = here->BSIM4v6Igcd; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_GM:
            value->rValue = here->BSIM4v6gm; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_GDS:
            value->rValue = here->BSIM4v6gds; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_GMBS:
            value->rValue = here->BSIM4v6gmbs; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_GBD:
            value->rValue = here->BSIM4v6gbd; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_GBS:
            value->rValue = here->BSIM4v6gbs; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
/*        case BSIM4v6_QB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v6qb); 
            return(OK); */
        case BSIM4v6_CQB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v6cqb); 
            return(OK);
/*        case BSIM4v6_QG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v6qg); 
            return(OK); */
        case BSIM4v6_CQG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v6cqg); 
            return(OK);
/*        case BSIM4v6_QD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v6qd); 
            return(OK); */
        case BSIM4v6_CQD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v6cqd); 
            return(OK);
/*        case BSIM4v6_QS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v6qs); 
            return(OK); */
        case BSIM4v6_QB:
            value->rValue = here->BSIM4v6qbulk; 
            value->rValue *= here->BSIM4v6m;
            return(OK); 
        case BSIM4v6_QG:
            value->rValue = here->BSIM4v6qgate; 
            value->rValue *= here->BSIM4v6m;
            return(OK); 
        case BSIM4v6_QS:
            value->rValue = here->BSIM4v6qsrc; 
            value->rValue *= here->BSIM4v6m;
            return(OK); 
        case BSIM4v6_QD:
            value->rValue = here->BSIM4v6qdrn; 
            value->rValue *= here->BSIM4v6m;
            return(OK); 
        case BSIM4v6_QDEF:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v6qdef); 
            return(OK); 
        case BSIM4v6_GCRG:
            value->rValue = here->BSIM4v6gcrg;
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_GTAU:
            value->rValue = here->BSIM4v6gtau;
            return(OK);
        case BSIM4v6_CGGB:
            value->rValue = here->BSIM4v6cggb; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_CGDB:
            value->rValue = here->BSIM4v6cgdb;
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_CGSB:
            value->rValue = here->BSIM4v6cgsb;
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_CDGB:
            value->rValue = here->BSIM4v6cdgb; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_CDDB:
            value->rValue = here->BSIM4v6cddb; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_CDSB:
            value->rValue = here->BSIM4v6cdsb; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_CBGB:
            value->rValue = here->BSIM4v6cbgb;
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_CBDB:
            value->rValue = here->BSIM4v6cbdb;
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_CBSB:
            value->rValue = here->BSIM4v6cbsb;
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_CSGB:
            value->rValue = here->BSIM4v6csgb;
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_CSDB:
            value->rValue = here->BSIM4v6csdb;
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_CSSB:
            value->rValue = here->BSIM4v6cssb;
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_CGBB:
            value->rValue = here->BSIM4v6cgbb;
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_CDBB:
            value->rValue = here->BSIM4v6cdbb;
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_CSBB:
            value->rValue = here->BSIM4v6csbb;
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_CBBB:
            value->rValue = here->BSIM4v6cbbb;
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_CAPBD:
            value->rValue = here->BSIM4v6capbd; 
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_CAPBS:
            value->rValue = here->BSIM4v6capbs;
            value->rValue *= here->BSIM4v6m;
            return(OK);
        case BSIM4v6_VON:
            value->rValue = here->BSIM4v6von; 
            return(OK);
        case BSIM4v6_VDSAT:
            value->rValue = here->BSIM4v6vdsat; 
            return(OK);
        case BSIM4v6_QBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v6qbs); 
            return(OK);
        case BSIM4v6_QBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v6qbd); 
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

