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

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "bsim4v4def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BSIM4v4ask(
CKTcircuit *ckt,
GENinstance *inst,
int which,
IFvalue *value,
IFvalue *select)
{
BSIM4v4instance *here = (BSIM4v4instance*)inst;

    NG_IGNORE(select);

    switch(which)
    {   case BSIM4v4_L:
            value->rValue = here->BSIM4v4l;
            return(OK);
        case BSIM4v4_W:
            value->rValue = here->BSIM4v4w;
            return(OK);
        case BSIM4v4_NF:
            value->rValue = here->BSIM4v4nf;
            return(OK);
        case BSIM4v4_MIN:
            value->iValue = here->BSIM4v4min;
            return(OK);
        case BSIM4v4_AS:
            value->rValue = here->BSIM4v4sourceArea;
            return(OK);
        case BSIM4v4_AD:
            value->rValue = here->BSIM4v4drainArea;
            return(OK);
        case BSIM4v4_PS:
            value->rValue = here->BSIM4v4sourcePerimeter;
            return(OK);
        case BSIM4v4_PD:
            value->rValue = here->BSIM4v4drainPerimeter;
            return(OK);
        case BSIM4v4_NRS:
            value->rValue = here->BSIM4v4sourceSquares;
            return(OK);
        case BSIM4v4_NRD:
            value->rValue = here->BSIM4v4drainSquares;
            return(OK);
        case BSIM4v4_OFF:
            value->rValue = here->BSIM4v4off;
            return(OK);
        case BSIM4v4_SA:
            value->rValue = here->BSIM4v4sa ;
            return(OK);
        case BSIM4v4_SB:
            value->rValue = here->BSIM4v4sb ;
            return(OK);
        case BSIM4v4_SD:
            value->rValue = here->BSIM4v4sd ;
            return(OK);
        case BSIM4v4_RBSB:
            value->rValue = here->BSIM4v4rbsb;
            return(OK);
        case BSIM4v4_RBDB:
            value->rValue = here->BSIM4v4rbdb;
            return(OK);
        case BSIM4v4_RBPB:
            value->rValue = here->BSIM4v4rbpb;
            return(OK);
        case BSIM4v4_RBPS:
            value->rValue = here->BSIM4v4rbps;
            return(OK);
        case BSIM4v4_RBPD:
            value->rValue = here->BSIM4v4rbpd;
            return(OK);
        case BSIM4v4_TRNQSMOD:
            value->iValue = here->BSIM4v4trnqsMod;
            return(OK);
        case BSIM4v4_ACNQSMOD:
            value->iValue = here->BSIM4v4acnqsMod;
            return(OK);
        case BSIM4v4_RBODYMOD:
            value->iValue = here->BSIM4v4rbodyMod;
            return(OK);
        case BSIM4v4_RGATEMOD:
            value->iValue = here->BSIM4v4rgateMod;
            return(OK);
        case BSIM4v4_GEOMOD:
            value->iValue = here->BSIM4v4geoMod;
            return(OK);
        case BSIM4v4_RGEOMOD:
            value->iValue = here->BSIM4v4rgeoMod;
            return(OK);
        case BSIM4v4_IC_VDS:
            value->rValue = here->BSIM4v4icVDS;
            return(OK);
        case BSIM4v4_IC_VGS:
            value->rValue = here->BSIM4v4icVGS;
            return(OK);
        case BSIM4v4_IC_VBS:
            value->rValue = here->BSIM4v4icVBS;
            return(OK);
        case BSIM4v4_DNODE:
            value->iValue = here->BSIM4v4dNode;
            return(OK);
        case BSIM4v4_GNODEEXT:
            value->iValue = here->BSIM4v4gNodeExt;
            return(OK);
        case BSIM4v4_SNODE:
            value->iValue = here->BSIM4v4sNode;
            return(OK);
        case BSIM4v4_BNODE:
            value->iValue = here->BSIM4v4bNode;
            return(OK);
        case BSIM4v4_DNODEPRIME:
            value->iValue = here->BSIM4v4dNodePrime;
            return(OK);
        case BSIM4v4_GNODEPRIME:
            value->iValue = here->BSIM4v4gNodePrime;
            return(OK);
        case BSIM4v4_GNODEMID:
            value->iValue = here->BSIM4v4gNodeMid;
            return(OK);
        case BSIM4v4_SNODEPRIME:
            value->iValue = here->BSIM4v4sNodePrime;
            return(OK);
        case BSIM4v4_DBNODE:
            value->iValue = here->BSIM4v4dbNode;
            return(OK);
        case BSIM4v4_BNODEPRIME:
            value->iValue = here->BSIM4v4bNodePrime;
            return(OK);
        case BSIM4v4_SBNODE:
            value->iValue = here->BSIM4v4sbNode;
            return(OK);
        case BSIM4v4_SOURCECONDUCT:
            value->rValue = here->BSIM4v4sourceConductance;
            return(OK);
        case BSIM4v4_DRAINCONDUCT:
            value->rValue = here->BSIM4v4drainConductance;
            return(OK);
        case BSIM4v4_VBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v4vbd);
            return(OK);
        case BSIM4v4_VBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v4vbs);
            return(OK);
        case BSIM4v4_VGS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v4vgs);
            return(OK);
        case BSIM4v4_VDS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v4vds);
            return(OK);
        case BSIM4v4_CD:
            value->rValue = here->BSIM4v4cd;
            return(OK);
        case BSIM4v4_CBS:
            value->rValue = here->BSIM4v4cbs;
            return(OK);
        case BSIM4v4_CBD:
            value->rValue = here->BSIM4v4cbd;
            return(OK);
        case BSIM4v4_CSUB:
            value->rValue = here->BSIM4v4csub;
            return(OK);
        case BSIM4v4_IGIDL:
            value->rValue = here->BSIM4v4Igidl;
            return(OK);
        case BSIM4v4_IGISL:
            value->rValue = here->BSIM4v4Igisl;
            return(OK);
        case BSIM4v4_IGS:
            value->rValue = here->BSIM4v4Igs;
            return(OK);
        case BSIM4v4_IGD:
            value->rValue = here->BSIM4v4Igd;
            return(OK);
        case BSIM4v4_IGB:
            value->rValue = here->BSIM4v4Igb;
            return(OK);
        case BSIM4v4_IGCS:
            value->rValue = here->BSIM4v4Igcs;
            return(OK);
        case BSIM4v4_IGCD:
            value->rValue = here->BSIM4v4Igcd;
            return(OK);
        case BSIM4v4_GM:
            value->rValue = here->BSIM4v4gm;
            return(OK);
        case BSIM4v4_GDS:
            value->rValue = here->BSIM4v4gds;
            return(OK);
        case BSIM4v4_GMBS:
            value->rValue = here->BSIM4v4gmbs;
            return(OK);
        case BSIM4v4_GBD:
            value->rValue = here->BSIM4v4gbd;
            return(OK);
        case BSIM4v4_GBS:
            value->rValue = here->BSIM4v4gbs;
            return(OK);
/*        case BSIM4v4_QB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v4qb);
            return(OK); */
        case BSIM4v4_CQB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v4cqb);
            return(OK);
/*        case BSIM4v4_QG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v4qg);
            return(OK); */
        case BSIM4v4_CQG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v4cqg);
            return(OK);
/*        case BSIM4v4_QD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v4qd);
            return(OK); */
        case BSIM4v4_CQD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v4cqd);
            return(OK);
/*        case BSIM4v4_QS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v4qs);
            return(OK); */
        case BSIM4v4_QB:
            value->rValue = here->BSIM4v4qbulk;
            return(OK);
        case BSIM4v4_QG:
            value->rValue = here->BSIM4v4qgate;
            return(OK);
        case BSIM4v4_QS:
            value->rValue = here->BSIM4v4qsrc;
            return(OK);
        case BSIM4v4_QD:
            value->rValue = here->BSIM4v4qdrn;
            return(OK);
        case BSIM4v4_CGGB:
            value->rValue = here->BSIM4v4cggb;
            return(OK);
        case BSIM4v4_CGDB:
            value->rValue = here->BSIM4v4cgdb;
            return(OK);
        case BSIM4v4_CGSB:
            value->rValue = here->BSIM4v4cgsb;
            return(OK);
        case BSIM4v4_CDGB:
            value->rValue = here->BSIM4v4cdgb;
            return(OK);
        case BSIM4v4_CDDB:
            value->rValue = here->BSIM4v4cddb;
            return(OK);
        case BSIM4v4_CDSB:
            value->rValue = here->BSIM4v4cdsb;
            return(OK);
        case BSIM4v4_CBGB:
            value->rValue = here->BSIM4v4cbgb;
            return(OK);
        case BSIM4v4_CBDB:
            value->rValue = here->BSIM4v4cbdb;
            return(OK);
        case BSIM4v4_CBSB:
            value->rValue = here->BSIM4v4cbsb;
            return(OK);
        case BSIM4v4_CSGB:
            value->rValue = here->BSIM4v4csgb;
            return(OK);
        case BSIM4v4_CSDB:
            value->rValue = here->BSIM4v4csdb;
            return(OK);
        case BSIM4v4_CSSB:
            value->rValue = here->BSIM4v4cssb;
            return(OK);
        case BSIM4v4_CGBB:
            value->rValue = here->BSIM4v4cgbb;
            return(OK);
        case BSIM4v4_CDBB:
            value->rValue = here->BSIM4v4cdbb;
            return(OK);
        case BSIM4v4_CSBB:
            value->rValue = here->BSIM4v4csbb;
            return(OK);
        case BSIM4v4_CBBB:
            value->rValue = here->BSIM4v4cbbb;
            return(OK);
        case BSIM4v4_CAPBD:
            value->rValue = here->BSIM4v4capbd;
            return(OK);
        case BSIM4v4_CAPBS:
            value->rValue = here->BSIM4v4capbs;
            return(OK);
        case BSIM4v4_VON:
            value->rValue = here->BSIM4v4von;
            return(OK);
        case BSIM4v4_VDSAT:
            value->rValue = here->BSIM4v4vdsat;
            return(OK);
        case BSIM4v4_QBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v4qbs);
            return(OK);
        case BSIM4v4_QBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM4v4qbd);
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

