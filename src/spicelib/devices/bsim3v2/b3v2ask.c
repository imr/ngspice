/* $Id$  */
/*
 $Log$
 Revision 1.1  2000-04-27 20:03:59  pnenzi
 Initial revision

 * Revision 3.2 1998/6/16  18:00:00  Weidong 
 * BSIM3v3.2 release
 *
*/
static char rcsid[] = "$Id$";

/*************************************/

/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v2ask.c
**********/

#include "ngspice.h"
#include <stdio.h>
#include <math.h>
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "bsim3v2def.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3V2ask(ckt,inst,which,value,select)
CKTcircuit *ckt;
GENinstance *inst;
int which;
IFvalue *value;
IFvalue *select;
{
BSIM3V2instance *here = (BSIM3V2instance*)inst;

    switch(which) 
    {   case BSIM3V2_L:
            value->rValue = here->BSIM3V2l;
            return(OK);
        case BSIM3V2_W:
            value->rValue = here->BSIM3V2w;
            return(OK);
        case BSIM3V2_AS:
            value->rValue = here->BSIM3V2sourceArea;
            return(OK);
        case BSIM3V2_AD:
            value->rValue = here->BSIM3V2drainArea;
            return(OK);
        case BSIM3V2_PS:
            value->rValue = here->BSIM3V2sourcePerimeter;
            return(OK);
        case BSIM3V2_PD:
            value->rValue = here->BSIM3V2drainPerimeter;
            return(OK);
        case BSIM3V2_NRS:
            value->rValue = here->BSIM3V2sourceSquares;
            return(OK);
        case BSIM3V2_NRD:
            value->rValue = here->BSIM3V2drainSquares;
            return(OK);
        case BSIM3V2_OFF:
            value->rValue = here->BSIM3V2off;
            return(OK);
        case BSIM3V2_NQSMOD:
            value->iValue = here->BSIM3V2nqsMod;
            return(OK);
        case BSIM3V2_IC_VBS:
            value->rValue = here->BSIM3V2icVBS;
            return(OK);
        case BSIM3V2_IC_VDS:
            value->rValue = here->BSIM3V2icVDS;
            return(OK);
        case BSIM3V2_IC_VGS:
            value->rValue = here->BSIM3V2icVGS;
            return(OK);
        case BSIM3V2_DNODE:
            value->iValue = here->BSIM3V2dNode;
            return(OK);
        case BSIM3V2_GNODE:
            value->iValue = here->BSIM3V2gNode;
            return(OK);
        case BSIM3V2_SNODE:
            value->iValue = here->BSIM3V2sNode;
            return(OK);
        case BSIM3V2_BNODE:
            value->iValue = here->BSIM3V2bNode;
            return(OK);
        case BSIM3V2_DNODEPRIME:
            value->iValue = here->BSIM3V2dNodePrime;
            return(OK);
        case BSIM3V2_SNODEPRIME:
            value->iValue = here->BSIM3V2sNodePrime;
            return(OK);
        case BSIM3V2_SOURCECONDUCT:
            value->rValue = here->BSIM3V2sourceConductance;
            return(OK);
        case BSIM3V2_DRAINCONDUCT:
            value->rValue = here->BSIM3V2drainConductance;
            return(OK);
        case BSIM3V2_VBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V2vbd);
            return(OK);
        case BSIM3V2_VBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V2vbs);
            return(OK);
        case BSIM3V2_VGS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V2vgs);
            return(OK);
        case BSIM3V2_VDS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V2vds);
            return(OK);
        case BSIM3V2_CD:
            value->rValue = here->BSIM3V2cd; 
            return(OK);
        case BSIM3V2_CBS:
            value->rValue = here->BSIM3V2cbs; 
            return(OK);
        case BSIM3V2_CBD:
            value->rValue = here->BSIM3V2cbd; 
            return(OK);
        case BSIM3V2_GM:
            value->rValue = here->BSIM3V2gm; 
            return(OK);
        case BSIM3V2_GDS:
            value->rValue = here->BSIM3V2gds; 
            return(OK);
        case BSIM3V2_GMBS:
            value->rValue = here->BSIM3V2gmbs; 
            return(OK);
        case BSIM3V2_GBD:
            value->rValue = here->BSIM3V2gbd; 
            return(OK);
        case BSIM3V2_GBS:
            value->rValue = here->BSIM3V2gbs; 
            return(OK);
        case BSIM3V2_QB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V2qb); 
            return(OK);
        case BSIM3V2_CQB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V2cqb); 
            return(OK);
        case BSIM3V2_QG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V2qg); 
            return(OK);
        case BSIM3V2_CQG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V2cqg); 
            return(OK);
        case BSIM3V2_QD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V2qd); 
            return(OK);
        case BSIM3V2_CQD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V2cqd); 
            return(OK);
        case BSIM3V2_CGG:
            value->rValue = here->BSIM3V2cggb; 
            return(OK);
        case BSIM3V2_CGD:
            value->rValue = here->BSIM3V2cgdb;
            return(OK);
        case BSIM3V2_CGS:
            value->rValue = here->BSIM3V2cgsb;
            return(OK);
        case BSIM3V2_CDG:
            value->rValue = here->BSIM3V2cdgb; 
            return(OK);
        case BSIM3V2_CDD:
            value->rValue = here->BSIM3V2cddb; 
            return(OK);
        case BSIM3V2_CDS:
            value->rValue = here->BSIM3V2cdsb; 
            return(OK);
        case BSIM3V2_CBG:
            value->rValue = here->BSIM3V2cbgb;
            return(OK);
        case BSIM3V2_CBDB:
            value->rValue = here->BSIM3V2cbdb;
            return(OK);
        case BSIM3V2_CBSB:
            value->rValue = here->BSIM3V2cbsb;
            return(OK);
        case BSIM3V2_CAPBD:
            value->rValue = here->BSIM3V2capbd; 
            return(OK);
        case BSIM3V2_CAPBS:
            value->rValue = here->BSIM3V2capbs;
            return(OK);
        case BSIM3V2_VON:
            value->rValue = here->BSIM3V2von; 
            return(OK);
        case BSIM3V2_VDSAT:
            value->rValue = here->BSIM3V2vdsat; 
            return(OK);
        case BSIM3V2_QBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V2qbs); 
            return(OK);
        case BSIM3V2_QBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3V2qbd); 
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

