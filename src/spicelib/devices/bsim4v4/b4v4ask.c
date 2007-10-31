/**** BSIM4.4.0  Released by Xuemei (Jane) Xi 03/04/2004 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b4ask.c of BSIM4.4.0.
 * Author: 2000 Weidong Liu
 * Authors: 2001- Xuemei Xi, Jin He, Kanyu Cao, Mohan Dunga, Mansun Chan, Ali Niknejad, Chenming Hu.
 * Project Director: Prof. Chenming Hu.
 * Modified by Xuemei Xi, 04/06/2001.
 * Modified by Xuemei Xi, 10/05/2001.
 * Modified by Xuemei Xi, 05/09/2003.
 **********/

#include "ngspice.h"
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "bsim4v4def.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM4V4ask(ckt,inst,which,value,select)
CKTcircuit *ckt;
GENinstance *inst;
int which;
IFvalue *value;
IFvalue *select;
{
BSIM4V4instance *here = (BSIM4V4instance*)inst;

    switch(which) 
    {   case BSIM4V4_L:
            value->rValue = here->BSIM4V4l;
            return(OK);
        case BSIM4V4_W:
            value->rValue = here->BSIM4V4w;
            return(OK);
        case BSIM4V4_M:
            value->rValue = here->BSIM4V4m;
            return(OK);
        case BSIM4V4_NF:
            value->rValue = here->BSIM4V4nf;
            return(OK);
        case BSIM4V4_MIN:
            value->iValue = here->BSIM4V4min;
            return(OK);
        case BSIM4V4_AS:
            value->rValue = here->BSIM4V4sourceArea;
            return(OK);
        case BSIM4V4_AD:
            value->rValue = here->BSIM4V4drainArea;
            return(OK);
        case BSIM4V4_PS:
            value->rValue = here->BSIM4V4sourcePerimeter;
            return(OK);
        case BSIM4V4_PD:
            value->rValue = here->BSIM4V4drainPerimeter;
            return(OK);
        case BSIM4V4_NRS:
            value->rValue = here->BSIM4V4sourceSquares;
            return(OK);
        case BSIM4V4_NRD:
            value->rValue = here->BSIM4V4drainSquares;
            return(OK);
        case BSIM4V4_OFF:
            value->rValue = here->BSIM4V4off;
            return(OK);
        case BSIM4V4_SA:
            value->rValue = here->BSIM4V4sa ;
            return(OK);
        case BSIM4V4_SB:
            value->rValue = here->BSIM4V4sb ;
            return(OK);
        case BSIM4V4_SD:
            value->rValue = here->BSIM4V4sd ;
            return(OK);
        case BSIM4V4_RBSB:
            value->rValue = here->BSIM4V4rbsb;
            return(OK);
        case BSIM4V4_RBDB:
            value->rValue = here->BSIM4V4rbdb;
            return(OK);
        case BSIM4V4_RBPB:
            value->rValue = here->BSIM4V4rbpb;
            return(OK);
        case BSIM4V4_RBPS:
            value->rValue = here->BSIM4V4rbps;
            return(OK);
        case BSIM4V4_RBPD:
            value->rValue = here->BSIM4V4rbpd;
            return(OK);
        case BSIM4V4_TRNQSMOD:
            value->iValue = here->BSIM4V4trnqsMod;
            return(OK);
        case BSIM4V4_ACNQSMOD:
            value->iValue = here->BSIM4V4acnqsMod;
            return(OK);
        case BSIM4V4_RBODYMOD:
            value->iValue = here->BSIM4V4rbodyMod;
            return(OK);
        case BSIM4V4_RGATEMOD:
            value->iValue = here->BSIM4V4rgateMod;
            return(OK);
        case BSIM4V4_GEOMOD:
            value->iValue = here->BSIM4V4geoMod;
            return(OK);
        case BSIM4V4_RGEOMOD:
            value->iValue = here->BSIM4V4rgeoMod;
            return(OK);
        case BSIM4V4_IC_VDS:
            value->rValue = here->BSIM4V4icVDS;
            return(OK);
        case BSIM4V4_IC_VGS:
            value->rValue = here->BSIM4V4icVGS;
            return(OK);
        case BSIM4V4_IC_VBS:
            value->rValue = here->BSIM4V4icVBS;
            return(OK);
        case BSIM4V4_DNODE:
            value->iValue = here->BSIM4V4dNode;
            return(OK);
        case BSIM4V4_GNODEEXT:
            value->iValue = here->BSIM4V4gNodeExt;
            return(OK);
        case BSIM4V4_SNODE:
            value->iValue = here->BSIM4V4sNode;
            return(OK);
        case BSIM4V4_BNODE:
            value->iValue = here->BSIM4V4bNode;
            return(OK);
        case BSIM4V4_DNODEPRIME:
            value->iValue = here->BSIM4V4dNodePrime;
            return(OK);
        case BSIM4V4_GNODEPRIME:
            value->iValue = here->BSIM4V4gNodePrime;
            return(OK);
        case BSIM4V4_GNODEMID:
            value->iValue = here->BSIM4V4gNodeMid;
            return(OK);
        case BSIM4V4_SNODEPRIME:
            value->iValue = here->BSIM4V4sNodePrime;
            return(OK);
        case BSIM4V4_DBNODE:
            value->iValue = here->BSIM4V4dbNode;
            return(OK);
        case BSIM4V4_BNODEPRIME:
            value->iValue = here->BSIM4V4bNodePrime;
            return(OK);
        case BSIM4V4_SBNODE:
            value->iValue = here->BSIM4V4sbNode;
            return(OK);
        case BSIM4V4_SOURCECONDUCT:
            value->rValue = here->BSIM4V4sourceConductance;
            return(OK);
        case BSIM4V4_DRAINCONDUCT:
            value->rValue = here->BSIM4V4drainConductance;
            return(OK);
        case BSIM4V4_VBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4V4vbd);
            return(OK);
        case BSIM4V4_VBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4V4vbs);
            return(OK);
        case BSIM4V4_VGS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4V4vgs);
            return(OK);
        case BSIM4V4_VDS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4V4vds);
            return(OK);
        case BSIM4V4_CD:
            value->rValue = here->BSIM4V4cd; 
            return(OK);
        case BSIM4V4_CBS:
            value->rValue = here->BSIM4V4cbs; 
            return(OK);
        case BSIM4V4_CBD:
            value->rValue = here->BSIM4V4cbd; 
            return(OK);
        case BSIM4V4_CSUB:
            value->rValue = here->BSIM4V4csub; 
            return(OK);
        case BSIM4V4_IGIDL:
            value->rValue = here->BSIM4V4Igidl; 
            return(OK);
        case BSIM4V4_IGISL:
            value->rValue = here->BSIM4V4Igisl; 
            return(OK);
        case BSIM4V4_IGS:
            value->rValue = here->BSIM4V4Igs; 
            return(OK);
        case BSIM4V4_IGD:
            value->rValue = here->BSIM4V4Igd; 
            return(OK);
        case BSIM4V4_IGB:
            value->rValue = here->BSIM4V4Igb; 
            return(OK);
        case BSIM4V4_IGCS:
            value->rValue = here->BSIM4V4Igcs; 
            return(OK);
        case BSIM4V4_IGCD:
            value->rValue = here->BSIM4V4Igcd; 
            return(OK);
        case BSIM4V4_GM:
            value->rValue = here->BSIM4V4gm; 
            return(OK);
        case BSIM4V4_GDS:
            value->rValue = here->BSIM4V4gds; 
            return(OK);
        case BSIM4V4_GMBS:
            value->rValue = here->BSIM4V4gmbs; 
            return(OK);
        case BSIM4V4_GBD:
            value->rValue = here->BSIM4V4gbd; 
            return(OK);
        case BSIM4V4_GBS:
            value->rValue = here->BSIM4V4gbs; 
            return(OK);
/*        case BSIM4V4_QB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4V4qb); 
            return(OK); */
        case BSIM4V4_CQB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4V4cqb); 
            return(OK);
/*        case BSIM4V4_QG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4V4qg); 
            return(OK); */
        case BSIM4V4_CQG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4V4cqg); 
            return(OK);
/*        case BSIM4V4_QD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4V4qd); 
            return(OK); */
        case BSIM4V4_CQD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4V4cqd); 
            return(OK);
/*        case BSIM4V4_QS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4V4qs); 
            return(OK); */
        case BSIM4V4_QB:
            value->rValue = here->BSIM4V4qbulk; 
            return(OK); 
        case BSIM4V4_QG:
            value->rValue = here->BSIM4V4qgate; 
            return(OK); 
        case BSIM4V4_QS:
            value->rValue = here->BSIM4V4qsrc; 
            return(OK); 
        case BSIM4V4_QD:
            value->rValue = here->BSIM4V4qdrn; 
            return(OK); 
        case BSIM4V4_CGGB:
            value->rValue = here->BSIM4V4cggb; 
            return(OK);
        case BSIM4V4_CGDB:
            value->rValue = here->BSIM4V4cgdb;
            return(OK);
        case BSIM4V4_CGSB:
            value->rValue = here->BSIM4V4cgsb;
            return(OK);
        case BSIM4V4_CDGB:
            value->rValue = here->BSIM4V4cdgb; 
            return(OK);
        case BSIM4V4_CDDB:
            value->rValue = here->BSIM4V4cddb; 
            return(OK);
        case BSIM4V4_CDSB:
            value->rValue = here->BSIM4V4cdsb; 
            return(OK);
        case BSIM4V4_CBGB:
            value->rValue = here->BSIM4V4cbgb;
            return(OK);
        case BSIM4V4_CBDB:
            value->rValue = here->BSIM4V4cbdb;
            return(OK);
        case BSIM4V4_CBSB:
            value->rValue = here->BSIM4V4cbsb;
            return(OK);
        case BSIM4V4_CSGB:
            value->rValue = here->BSIM4V4csgb;
            return(OK);
        case BSIM4V4_CSDB:
            value->rValue = here->BSIM4V4csdb;
            return(OK);
        case BSIM4V4_CSSB:
            value->rValue = here->BSIM4V4cssb;
            return(OK);
        case BSIM4V4_CGBB:
            value->rValue = here->BSIM4V4cgbb;
            return(OK);
        case BSIM4V4_CDBB:
            value->rValue = here->BSIM4V4cdbb;
            return(OK);
        case BSIM4V4_CSBB:
            value->rValue = here->BSIM4V4csbb;
            return(OK);
        case BSIM4V4_CBBB:
            value->rValue = here->BSIM4V4cbbb;
            return(OK);
        case BSIM4V4_CAPBD:
            value->rValue = here->BSIM4V4capbd; 
            return(OK);
        case BSIM4V4_CAPBS:
            value->rValue = here->BSIM4V4capbs;
            return(OK);
        case BSIM4V4_VON:
            value->rValue = here->BSIM4V4von; 
            return(OK);
        case BSIM4V4_VDSAT:
            value->rValue = here->BSIM4V4vdsat; 
            return(OK);
        case BSIM4V4_QBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4V4qbs); 
            return(OK);
        case BSIM4V4_QBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4V4qbd); 
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

