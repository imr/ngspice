/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Author: 1997-1999 Weidong Liu.
File: b3ask.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "bsim3def.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3ask(ckt,inst,which,value,select)
CKTcircuit *ckt;
GENinstance *inst;
int which;
IFvalue *value;
IFvalue *select;
{
BSIM3instance *here = (BSIM3instance*)inst;

    switch(which) 
    {   case BSIM3_L:
            value->rValue = here->BSIM3l;
            return(OK);
        case BSIM3_W:
            value->rValue = here->BSIM3w;
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
            return(OK);
        case BSIM3_DRAINCONDUCT:
            value->rValue = here->BSIM3drainConductance;
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
            return(OK);
        case BSIM3_CBS:
            value->rValue = here->BSIM3cbs; 
            return(OK);
        case BSIM3_CBD:
            value->rValue = here->BSIM3cbd; 
            return(OK);
        case BSIM3_GM:
            value->rValue = here->BSIM3gm; 
            return(OK);
        case BSIM3_GDS:
            value->rValue = here->BSIM3gds; 
            return(OK);
        case BSIM3_GMBS:
            value->rValue = here->BSIM3gmbs; 
            return(OK);
        case BSIM3_GBD:
            value->rValue = here->BSIM3gbd; 
            return(OK);
        case BSIM3_GBS:
            value->rValue = here->BSIM3gbs; 
            return(OK);
        case BSIM3_QB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3qb); 
            return(OK);
        case BSIM3_CQB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3cqb); 
            return(OK);
        case BSIM3_QG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3qg); 
            return(OK);
        case BSIM3_CQG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3cqg); 
            return(OK);
        case BSIM3_QD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3qd); 
            return(OK);
        case BSIM3_CQD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3cqd); 
            return(OK);
        case BSIM3_CGG:
            value->rValue = here->BSIM3cggb; 
            return(OK);
        case BSIM3_CGD:
            value->rValue = here->BSIM3cgdb;
            return(OK);
        case BSIM3_CGS:
            value->rValue = here->BSIM3cgsb;
            return(OK);
        case BSIM3_CDG:
            value->rValue = here->BSIM3cdgb; 
            return(OK);
        case BSIM3_CDD:
            value->rValue = here->BSIM3cddb; 
            return(OK);
        case BSIM3_CDS:
            value->rValue = here->BSIM3cdsb; 
            return(OK);
        case BSIM3_CBG:
            value->rValue = here->BSIM3cbgb;
            return(OK);
        case BSIM3_CBDB:
            value->rValue = here->BSIM3cbdb;
            return(OK);
        case BSIM3_CBSB:
            value->rValue = here->BSIM3cbsb;
            return(OK);
        case BSIM3_CAPBD:
            value->rValue = here->BSIM3capbd; 
            return(OK);
        case BSIM3_CAPBS:
            value->rValue = here->BSIM3capbs;
            return(OK);
        case BSIM3_VON:
            value->rValue = here->BSIM3von; 
            return(OK);
        case BSIM3_VDSAT:
            value->rValue = here->BSIM3vdsat; 
            return(OK);
        case BSIM3_QBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3qbs); 
            return(OK);
        case BSIM3_QBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3qbd); 
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

