/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1sask.c
**********/

#include "ngspice.h"
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "bsim3v1sdef.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3v1Sask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, 
            IFvalue *select)
{
BSIM3v1Sinstance *here = (BSIM3v1Sinstance*)inst;

    switch(which) 
    {   case BSIM3v1S_L:
            value->rValue = here->BSIM3v1Sl;
            return(OK);
        case BSIM3v1S_W:
            value->rValue = here->BSIM3v1Sw;
            return(OK);
        case BSIM3v1S_AS:
            value->rValue = here->BSIM3v1SsourceArea;
            return(OK);
        case BSIM3v1S_AD:
            value->rValue = here->BSIM3v1SdrainArea;
            return(OK);
        case BSIM3v1S_PS:
            value->rValue = here->BSIM3v1SsourcePerimeter;
            return(OK);
        case BSIM3v1S_PD:
            value->rValue = here->BSIM3v1SdrainPerimeter;
            return(OK);
        case BSIM3v1S_NRS:
            value->rValue = here->BSIM3v1SsourceSquares;
            return(OK);
        case BSIM3v1S_NRD:
            value->rValue = here->BSIM3v1SdrainSquares;
            return(OK);
        case BSIM3v1S_OFF:
            value->rValue = here->BSIM3v1Soff;
            return(OK);
        case BSIM3v1S_NQSMOD:
            value->iValue = here->BSIM3v1SnqsMod;
            return(OK);
        case BSIM3v1S_M:
            value->rValue = here->BSIM3v1Sm;
            return(OK);
        case BSIM3v1S_IC_VBS:
            value->rValue = here->BSIM3v1SicVBS;
            return(OK);
        case BSIM3v1S_IC_VDS:
            value->rValue = here->BSIM3v1SicVDS;
            return(OK);
        case BSIM3v1S_IC_VGS:
            value->rValue = here->BSIM3v1SicVGS;
            return(OK);
        case BSIM3v1S_DNODE:
            value->iValue = here->BSIM3v1SdNode;
            return(OK);
        case BSIM3v1S_GNODE:
            value->iValue = here->BSIM3v1SgNode;
            return(OK);
        case BSIM3v1S_SNODE:
            value->iValue = here->BSIM3v1SsNode;
            return(OK);
        case BSIM3v1S_BNODE:
            value->iValue = here->BSIM3v1SbNode;
            return(OK);
        case BSIM3v1S_DNODEPRIME:
            value->iValue = here->BSIM3v1SdNodePrime;
            return(OK);
        case BSIM3v1S_SNODEPRIME:
            value->iValue = here->BSIM3v1SsNodePrime;
            return(OK);
        case BSIM3v1S_SOURCECONDUCT:
            value->rValue = here->BSIM3v1SsourceConductance;
            return(OK);
        case BSIM3v1S_DRAINCONDUCT:
            value->rValue = here->BSIM3v1SdrainConductance;
            return(OK);
        case BSIM3v1S_VBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Svbd);
            return(OK);
        case BSIM3v1S_VBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Svbs);
            return(OK);
        case BSIM3v1S_VGS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Svgs);
            return(OK);
        case BSIM3v1S_VDS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Svds);
            return(OK);
        case BSIM3v1S_CD:
            value->rValue = here->BSIM3v1Scd; 
            return(OK);
        case BSIM3v1S_CBS:
            value->rValue = here->BSIM3v1Scbs; 
            return(OK);
        case BSIM3v1S_CBD:
            value->rValue = here->BSIM3v1Scbd; 
            return(OK);
        case BSIM3v1S_GM:
            value->rValue = here->BSIM3v1Sgm; 
            return(OK);
        case BSIM3v1S_GDS:
            value->rValue = here->BSIM3v1Sgds; 
            return(OK);
        case BSIM3v1S_GMBS:
            value->rValue = here->BSIM3v1Sgmbs; 
            return(OK);
        case BSIM3v1S_GBD:
            value->rValue = here->BSIM3v1Sgbd; 
            return(OK);
        case BSIM3v1S_GBS:
            value->rValue = here->BSIM3v1Sgbs; 
            return(OK);
        case BSIM3v1S_QB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Sqb); 
            return(OK);
        case BSIM3v1S_CQB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Scqb); 
            return(OK);
        case BSIM3v1S_QG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Sqg); 
            return(OK);
        case BSIM3v1S_CQG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Scqg); 
            return(OK);
        case BSIM3v1S_QD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Sqd); 
            return(OK);
        case BSIM3v1S_CQD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Scqd); 
            return(OK);
        case BSIM3v1S_CGG:
            value->rValue = here->BSIM3v1Scggb; 
            return(OK);
        case BSIM3v1S_CGD:
            value->rValue = here->BSIM3v1Scgdb;
            return(OK);
        case BSIM3v1S_CGS:
            value->rValue = here->BSIM3v1Scgsb;
            return(OK);
        case BSIM3v1S_CDG:
            value->rValue = here->BSIM3v1Scdgb; 
            return(OK);
        case BSIM3v1S_CDD:
            value->rValue = here->BSIM3v1Scddb; 
            return(OK);
        case BSIM3v1S_CDS:
            value->rValue = here->BSIM3v1Scdsb; 
            return(OK);
        case BSIM3v1S_CBG:
            value->rValue = here->BSIM3v1Scbgb;
            return(OK);
        case BSIM3v1S_CBDB:
            value->rValue = here->BSIM3v1Scbdb;
            return(OK);
        case BSIM3v1S_CBSB:
            value->rValue = here->BSIM3v1Scbsb;
            return(OK);
        case BSIM3v1S_CAPBD:
            value->rValue = here->BSIM3v1Scapbd; 
            return(OK);
        case BSIM3v1S_CAPBS:
            value->rValue = here->BSIM3v1Scapbs;
            return(OK);
        case BSIM3v1S_VON:
            value->rValue = here->BSIM3v1Svon; 
            return(OK);
        case BSIM3v1S_VDSAT:
            value->rValue = here->BSIM3v1Svdsat; 
            return(OK);
        case BSIM3v1S_QBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Sqbs); 
            return(OK);
        case BSIM3v1S_QBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Sqbd); 
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

