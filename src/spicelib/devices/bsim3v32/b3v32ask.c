/**** BSIM3v3.2.4, Released by Xuemei Xi 12/14/2001 ****/

/**********
 * Copyright 2001 Regents of the University of California. All rights reserved.
 * File: b3ask.c of BSIM3v3.2.4
 * Author: 1995 Min-Chie Jeng and Mansun Chan
 * Author: 1997-1999 Weidong Liu.
 * Author: 2001  Xuemei Xi
 * Modified bt Paolo Nenzi 2002 and Dietmar Warning 2003
 **********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "bsim3v32def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BSIM3v32ask (CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value,
          IFvalue *select)
{
BSIM3v32instance *here = (BSIM3v32instance*)inst;
BSIM3v32model *model = BSIM3v32modPtr(here); /* for lmlt */

    NG_IGNORE(select);

    switch(which)
    {   case BSIM3v32_L:
            value->rValue = here->BSIM3v32l;
            value->rValue *= model->BSIM3v32lmlt;
            return(OK);
        case BSIM3v32_W:
            value->rValue = here->BSIM3v32w;
            return(OK);
        case BSIM3v32_M:
            value->rValue = here->BSIM3v32m;
            return (OK);
        case BSIM3v32_AS:
            value->rValue = here->BSIM3v32sourceArea;
            return(OK);
        case BSIM3v32_AD:
            value->rValue = here->BSIM3v32drainArea;
            return(OK);
        case BSIM3v32_PS:
            value->rValue = here->BSIM3v32sourcePerimeter;
            return(OK);
        case BSIM3v32_PD:
            value->rValue = here->BSIM3v32drainPerimeter;
            return(OK);
        case BSIM3v32_NRS:
            value->rValue = here->BSIM3v32sourceSquares;
            return(OK);
        case BSIM3v32_NRD:
            value->rValue = here->BSIM3v32drainSquares;
            return(OK);
        case BSIM3v32_OFF:
            value->rValue = here->BSIM3v32off;
            return(OK);
        case BSIM3v32_NQSMOD:
            value->iValue = here->BSIM3v32nqsMod;
            return(OK);
        case BSIM3v32_GEO:
            value->iValue = here->BSIM3v32geo;
            return(OK);
        case BSIM3v32_DELVTO:
            value->rValue = here->BSIM3v32delvto;
            return(OK);
        case BSIM3v32_MULU0:
            value->rValue = here->BSIM3v32mulu0;
            return(OK);
        case BSIM3v32_IC_VBS:
            value->rValue = here->BSIM3v32icVBS;
            return(OK);
        case BSIM3v32_IC_VDS:
            value->rValue = here->BSIM3v32icVDS;
            return(OK);
        case BSIM3v32_IC_VGS:
            value->rValue = here->BSIM3v32icVGS;
            return(OK);
        case BSIM3v32_DNODE:
            value->iValue = here->BSIM3v32dNode;
            return(OK);
        case BSIM3v32_GNODE:
            value->iValue = here->BSIM3v32gNode;
            return(OK);
        case BSIM3v32_SNODE:
            value->iValue = here->BSIM3v32sNode;
            return(OK);
        case BSIM3v32_BNODE:
            value->iValue = here->BSIM3v32bNode;
            return(OK);
        case BSIM3v32_DNODEPRIME:
            value->iValue = here->BSIM3v32dNodePrime;
            return(OK);
        case BSIM3v32_SNODEPRIME:
            value->iValue = here->BSIM3v32sNodePrime;
            return(OK);
        case BSIM3v32_SOURCECONDUCT:
            value->rValue = here->BSIM3v32sourceConductance;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_DRAINCONDUCT:
            value->rValue = here->BSIM3v32drainConductance;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_VBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v32vbd);
            return(OK);
        case BSIM3v32_VBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v32vbs);
            return(OK);
        case BSIM3v32_VGS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v32vgs);
            return(OK);
        case BSIM3v32_VDS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v32vds);
            return(OK);
        case BSIM3v32_CD:
            value->rValue = here->BSIM3v32cd;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_CBS:
            value->rValue = here->BSIM3v32cbs;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_CBD:
            value->rValue = here->BSIM3v32cbd;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_GM:
            value->rValue = here->BSIM3v32gm;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_GDS:
            value->rValue = here->BSIM3v32gds;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_GMBS:
            value->rValue = here->BSIM3v32gmbs;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_GBD:
            value->rValue = here->BSIM3v32gbd;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_GBS:
            value->rValue = here->BSIM3v32gbs;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_QB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v32qb);
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_CQB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v32cqb);
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_QG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v32qg);
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_CQG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v32cqg);
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_QD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v32qd);
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_CQD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v32cqd);
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_CGG:
            value->rValue = here->BSIM3v32cggb;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_CGD:
            value->rValue = here->BSIM3v32cgdb;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_CGS:
            value->rValue = here->BSIM3v32cgsb;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_CDG:
            value->rValue = here->BSIM3v32cdgb;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_CDD:
            value->rValue = here->BSIM3v32cddb;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_CDS:
            value->rValue = here->BSIM3v32cdsb;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_CBG:
            value->rValue = here->BSIM3v32cbgb;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_CBDB:
            value->rValue = here->BSIM3v32cbdb;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_CBSB:
            value->rValue = here->BSIM3v32cbsb;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_CAPBD:
            value->rValue = here->BSIM3v32capbd;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_CAPBS:
            value->rValue = here->BSIM3v32capbs;
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_VON:
            value->rValue = here->BSIM3v32von;
            return(OK);
        case BSIM3v32_VDSAT:
            value->rValue = here->BSIM3v32vdsat;
            return(OK);
        case BSIM3v32_QBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v32qbs);
            value->rValue *= here->BSIM3v32m;
            return(OK);
        case BSIM3v32_QBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v32qbd);
            value->rValue *= here->BSIM3v32m;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

