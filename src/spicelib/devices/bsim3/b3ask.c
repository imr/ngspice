/**** BSIM3v3.3.0, Released by Xuemei Xi 07/29/2005 ****/

/**********
 * Copyright 2004 Regents of the University of California. All rights reserved.
 * File: b3ask.c of BSIM3v3.3.0
 * Author: 1995 Min-Chie Jeng and Mansun Chan
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "bsim3def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BSIM3ask(
CKTcircuit *ckt,
GENinstance *inst,
int which,
IFvalue *value,
IFvalue *select)
{
BSIM3instance *here = (BSIM3instance*)inst;

    NG_IGNORE(select);

    switch(which)
    {   case BSIM3_L:
            value->rValue = here->BSIM3l;
            return(OK);
        case BSIM3_W:
            value->rValue = here->BSIM3w;
            return(OK);
        case BSIM3_M:
            value->rValue = here->BSIM3m;
            return(OK);
        case BSIM3_AS:
            value->rValue = here->BSIM3sourceArea;
            return(OK);
        case BSIM3_AD:
            value->rValue = here->BSIM3drainArea;
            return(OK);
        case BSIM3_PS:
            value->rValue = here->BSIM3sourcePerimeter;
            return(OK);
        case BSIM3_PD:
            value->rValue = here->BSIM3drainPerimeter;
            return(OK);
        case BSIM3_NRS:
            value->rValue = here->BSIM3sourceSquares;
            return(OK);
        case BSIM3_NRD:
            value->rValue = here->BSIM3drainSquares;
            return(OK);
        case BSIM3_OFF:
            value->rValue = here->BSIM3off;
            return(OK);
        case BSIM3_NQSMOD:
            value->iValue = here->BSIM3nqsMod;
            return(OK);
        case BSIM3_ACNQSMOD:
            value->iValue = here->BSIM3acnqsMod;
            return(OK);
        case BSIM3_GEO:
            value->iValue = here->BSIM3geo;
            return(OK);
        case BSIM3_DELVTO:
            value->rValue = here->BSIM3delvto;
            return(OK);
        case BSIM3_MULU0:
            value->rValue = here->BSIM3mulu0;
            return(OK);
        case BSIM3_IC_VBS:
            value->rValue = here->BSIM3icVBS;
            return(OK);
        case BSIM3_IC_VDS:
            value->rValue = here->BSIM3icVDS;
            return(OK);
        case BSIM3_IC_VGS:
            value->rValue = here->BSIM3icVGS;
            return(OK);
        case BSIM3_DNODE:
            value->iValue = here->BSIM3dNode;
            return(OK);
        case BSIM3_GNODE:
            value->iValue = here->BSIM3gNode;
            return(OK);
        case BSIM3_SNODE:
            value->iValue = here->BSIM3sNode;
            return(OK);
        case BSIM3_BNODE:
            value->iValue = here->BSIM3bNode;
            return(OK);
        case BSIM3_DNODEPRIME:
            value->iValue = here->BSIM3dNodePrime;
            return(OK);
        case BSIM3_SNODEPRIME:
            value->iValue = here->BSIM3sNodePrime;
            return(OK);
        case BSIM3_SOURCECONDUCT:
            value->rValue = here->BSIM3sourceConductance;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_DRAINCONDUCT:
            value->rValue = here->BSIM3drainConductance;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_VBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3vbd);
            return(OK);
        case BSIM3_VBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3vbs);
            return(OK);
        case BSIM3_VGS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3vgs);
            return(OK);
        case BSIM3_VDS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3vds);
            return(OK);
        case BSIM3_CD:
            value->rValue = here->BSIM3cd;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_CBS:
            value->rValue = here->BSIM3cbs;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_CBD:
            value->rValue = here->BSIM3cbd;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_GM:
            value->rValue = here->BSIM3gm;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_GDS:
            value->rValue = here->BSIM3gds;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_GMBS:
            value->rValue = here->BSIM3gmbs;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_GBD:
            value->rValue = here->BSIM3gbd;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_GBS:
            value->rValue = here->BSIM3gbs;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_QB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3qb);
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_CQB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3cqb);
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_QG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3qg);
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_CQG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3cqg);
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_QD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3qd);
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_CQD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3cqd);
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_CGG:
            value->rValue = here->BSIM3cggb;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_CGD:
            value->rValue = here->BSIM3cgdb;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_CGS:
            value->rValue = here->BSIM3cgsb;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_CDG:
            value->rValue = here->BSIM3cdgb;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_CDD:
            value->rValue = here->BSIM3cddb;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_CDS:
            value->rValue = here->BSIM3cdsb;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_CBG:
            value->rValue = here->BSIM3cbgb;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_CBDB:
            value->rValue = here->BSIM3cbdb;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_CBSB:
            value->rValue = here->BSIM3cbsb;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_CAPBD:
            value->rValue = here->BSIM3capbd;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_CAPBS:
            value->rValue = here->BSIM3capbs;
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_VON:
            value->rValue = here->BSIM3von;
            return(OK);
        case BSIM3_VDSAT:
            value->rValue = here->BSIM3vdsat;
            return(OK);
        case BSIM3_QBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3qbs);
            value->rValue *= here->BSIM3m;
            return(OK);
        case BSIM3_QBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3qbd);
            value->rValue *= here->BSIM3m;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

