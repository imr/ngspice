/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v1ask.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "bsim3v1def.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3V1ask(ckt,inst,which,value,select)
CKTcircuit *ckt;
GENinstance *inst;
int which;
IFvalue *value;
IFvalue *select;
{
BSIM3V1instance *here = (BSIM3V1instance*)inst;

    switch(which) 
    {   case BSIM3V1_L:
            value->rValue = here->BSIM3V1l;
            return(OK);
        case BSIM3V1_W:
            value->rValue = here->BSIM3V1w;
            return(OK);
        case BSIM3V1_AS:
            value->rValue = here->BSIM3V1sourceArea;
            return(OK);
        case BSIM3V1_AD:
            value->rValue = here->BSIM3V1drainArea;
            return(OK);
        case BSIM3V1_PS:
            value->rValue = here->BSIM3V1sourcePerimeter;
            return(OK);
        case BSIM3V1_PD:
            value->rValue = here->BSIM3V1drainPerimeter;
            return(OK);
        case BSIM3V1_NRS:
            value->rValue = here->BSIM3V1sourceSquares;
            return(OK);
        case BSIM3V1_NRD:
            value->rValue = here->BSIM3V1drainSquares;
            return(OK);
        case BSIM3V1_OFF:
            value->rValue = here->BSIM3V1off;
            return(OK);
        case BSIM3V1_NQSMOD:
            value->iValue = here->BSIM3V1nqsMod;
            return(OK);
        case BSIM3V1_M:
            value->rValue = here->BSIM3V1m;
            return(OK);
        case BSIM3V1_IC_VBS:
            value->rValue = here->BSIM3V1icVBS;
            return(OK);
        case BSIM3V1_IC_VDS:
            value->rValue = here->BSIM3V1icVDS;
            return(OK);
        case BSIM3V1_IC_VGS:
            value->rValue = here->BSIM3V1icVGS;
            return(OK);
        case BSIM3V1_DNODE:
            value->iValue = here->BSIM3V1dNode;
            return(OK);
        case BSIM3V1_GNODE:
            value->iValue = here->BSIM3V1gNode;
            return(OK);
        case BSIM3V1_SNODE:
            value->iValue = here->BSIM3V1sNode;
            return(OK);
        case BSIM3V1_BNODE:
            value->iValue = here->BSIM3V1bNode;
            return(OK);
        case BSIM3V1_DNODEPRIME:
            value->iValue = here->BSIM3V1dNodePrime;
            return(OK);
        case BSIM3V1_SNODEPRIME:
            value->iValue = here->BSIM3V1sNodePrime;
            return(OK);
        case BSIM3V1_SOURCECONDUCT:
            value->rValue = here->BSIM3V1sourceConductance;
            return(OK);
        case BSIM3V1_DRAINCONDUCT:
            value->rValue = here->BSIM3V1drainConductance;
            return(OK);
        case BSIM3V1_VBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V1vbd);
            return(OK);
        case BSIM3V1_VBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V1vbs);
            return(OK);
        case BSIM3V1_VGS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V1vgs);
            return(OK);
        case BSIM3V1_VDS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V1vds);
            return(OK);
        case BSIM3V1_CD:
            value->rValue = here->BSIM3V1cd; 
            return(OK);
        case BSIM3V1_CBS:
            value->rValue = here->BSIM3V1cbs; 
            return(OK);
        case BSIM3V1_CBD:
            value->rValue = here->BSIM3V1cbd; 
            return(OK);
        case BSIM3V1_GM:
            value->rValue = here->BSIM3V1gm; 
            return(OK);
        case BSIM3V1_GDS:
            value->rValue = here->BSIM3V1gds; 
            return(OK);
        case BSIM3V1_GMBS:
            value->rValue = here->BSIM3V1gmbs; 
            return(OK);
        case BSIM3V1_GBD:
            value->rValue = here->BSIM3V1gbd; 
            return(OK);
        case BSIM3V1_GBS:
            value->rValue = here->BSIM3V1gbs; 
            return(OK);
        case BSIM3V1_QB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V1qb); 
            return(OK);
        case BSIM3V1_CQB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V1cqb); 
            return(OK);
        case BSIM3V1_QG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V1qg); 
            return(OK);
        case BSIM3V1_CQG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V1cqg); 
            return(OK);
        case BSIM3V1_QD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V1qd); 
            return(OK);
        case BSIM3V1_CQD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V1cqd); 
            return(OK);
        case BSIM3V1_CGG:
            value->rValue = here->BSIM3V1cggb; 
            return(OK);
        case BSIM3V1_CGD:
            value->rValue = here->BSIM3V1cgdb;
            return(OK);
        case BSIM3V1_CGS:
            value->rValue = here->BSIM3V1cgsb;
            return(OK);
        case BSIM3V1_CDG:
            value->rValue = here->BSIM3V1cdgb; 
            return(OK);
        case BSIM3V1_CDD:
            value->rValue = here->BSIM3V1cddb; 
            return(OK);
        case BSIM3V1_CDS:
            value->rValue = here->BSIM3V1cdsb; 
            return(OK);
        case BSIM3V1_CBG:
            value->rValue = here->BSIM3V1cbgb;
            return(OK);
        case BSIM3V1_CBDB:
            value->rValue = here->BSIM3V1cbdb;
            return(OK);
        case BSIM3V1_CBSB:
            value->rValue = here->BSIM3V1cbsb;
            return(OK);
        case BSIM3V1_CAPBD:
            value->rValue = here->BSIM3V1capbd; 
            return(OK);
        case BSIM3V1_CAPBS:
            value->rValue = here->BSIM3V1capbs;
            return(OK);
        case BSIM3V1_VON:
            value->rValue = here->BSIM3V1von; 
            return(OK);
        case BSIM3V1_VDSAT:
            value->rValue = here->BSIM3V1vdsat; 
            return(OK);
        case BSIM3V1_QBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V1qbs); 
            return(OK);
        case BSIM3V1_QBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V1qbd); 
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

