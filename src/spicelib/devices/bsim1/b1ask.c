/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Hong J. Park
**********/

#include "ngspice.h"
#include <stdio.h>
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "bsim1def.h"
#include "sperror.h"
#include "suffix.h"

/*ARGSUSED*/
int
B1ask(ckt,inst,which,value,select)
    CKTcircuit *ckt;
    GENinstance *inst;
    int which;
    IFvalue *value;
    IFvalue *select;
{
    B1instance *here = (B1instance*)inst;

    switch(which) {
        case BSIM1_L:
            value->rValue = here->B1l;
            return(OK);
        case BSIM1_W:
            value->rValue = here->B1w;
            return(OK);
        case BSIM1_AS:
            value->rValue = here->B1sourceArea;
            return(OK);
        case BSIM1_AD:
            value->rValue = here->B1drainArea;
            return(OK);
        case BSIM1_PS:
            value->rValue = here->B1sourcePerimeter;
            return(OK);
        case BSIM1_PD:
            value->rValue = here->B1drainPerimeter;
            return(OK);
        case BSIM1_NRS:
            value->rValue = here->B1sourceSquares;
            return(OK);
        case BSIM1_NRD:
            value->rValue = here->B1drainSquares;
            return(OK);
        case BSIM1_OFF:
            value->rValue = here->B1off;
            return(OK);
        case BSIM1_IC_VBS:
            value->rValue = here->B1icVBS;
            return(OK);
        case BSIM1_IC_VDS:
            value->rValue = here->B1icVDS;
            return(OK);
        case BSIM1_IC_VGS:
            value->rValue = here->B1icVGS;
            return(OK);
        case BSIM1_DNODE:
            value->iValue = here->B1dNode;
            return(OK);
        case BSIM1_GNODE:
            value->iValue = here->B1gNode;
            return(OK);
        case BSIM1_SNODE:
            value->iValue = here->B1sNode;
            return(OK);
        case BSIM1_BNODE:
            value->iValue = here->B1bNode;
            return(OK);
        case BSIM1_DNODEPRIME:
            value->iValue = here->B1dNodePrime;
            return(OK);
        case BSIM1_SNODEPRIME:
            value->iValue = here->B1sNodePrime;
            return(OK);
        case BSIM1_SOURCECONDUCT:
            value->rValue = here->B1sourceConductance;
            return(OK);
        case BSIM1_DRAINCONDUCT:
            value->rValue = here->B1drainConductance;
            return(OK);
        case BSIM1_VBD:
            value->rValue = *(ckt->CKTstate0 + here->B1vbd);
            return(OK);
        case BSIM1_VBS:
            value->rValue = *(ckt->CKTstate0 + here->B1vbs);
            return(OK);
        case BSIM1_VGS:
            value->rValue = *(ckt->CKTstate0 + here->B1vgs);
            return(OK);
        case BSIM1_VDS:
            value->rValue = *(ckt->CKTstate0 + here->B1vds);
            return(OK);
        case BSIM1_CD:
            value->rValue = *(ckt->CKTstate0 + here->B1cd); 
            return(OK);
        case BSIM1_CBS:
            value->rValue = *(ckt->CKTstate0 + here->B1cbs); 
            return(OK);
        case BSIM1_CBD:
            value->rValue = *(ckt->CKTstate0 + here->B1cbd); 
            return(OK);
        case BSIM1_GM:
            value->rValue = *(ckt->CKTstate0 + here->B1gm); 
            return(OK);
        case BSIM1_GDS:
            value->rValue = *(ckt->CKTstate0 + here->B1gds); 
            return(OK);
        case BSIM1_GMBS:
            value->rValue = *(ckt->CKTstate0 + here->B1gmbs); 
            return(OK);
        case BSIM1_GBD:
            value->rValue = *(ckt->CKTstate0 + here->B1gbd); 
            return(OK);
        case BSIM1_GBS:
            value->rValue = *(ckt->CKTstate0 + here->B1gbs); 
            return(OK);
        case BSIM1_QB:
            value->rValue = *(ckt->CKTstate0 + here->B1qb); 
            return(OK);
        case BSIM1_CQB:
            value->rValue = *(ckt->CKTstate0 + here->B1cqb); 
            return(OK);
        case BSIM1_QG:
            value->rValue = *(ckt->CKTstate0 + here->B1qg); 
            return(OK);
        case BSIM1_CQG:
            value->rValue = *(ckt->CKTstate0 + here->B1cqg); 
            return(OK);
        case BSIM1_QD:
            value->rValue = *(ckt->CKTstate0 + here->B1qd); 
            return(OK);
        case BSIM1_CQD:
            value->rValue = *(ckt->CKTstate0 + here->B1cqd); 
            return(OK);
        case BSIM1_CGG:
            value->rValue = *(ckt->CKTstate0 + here->B1cggb); 
            return(OK);
        case BSIM1_CGD:
            value->rValue = *(ckt->CKTstate0 + here->B1cgdb); 
            return(OK);
        case BSIM1_CGS:
            value->rValue = *(ckt->CKTstate0 + here->B1cgsb); 
            return(OK);
        case BSIM1_CBG:
            value->rValue = *(ckt->CKTstate0 + here->B1cbgb); 
            return(OK);
        case BSIM1_CAPBD:
            value->rValue = *(ckt->CKTstate0 + here->B1capbd); 
            return(OK);
        case BSIM1_CQBD:
            value->rValue = *(ckt->CKTstate0 + here->B1cqbd); 
            return(OK);
        case BSIM1_CAPBS:
            value->rValue = *(ckt->CKTstate0 + here->B1capbs); 
            return(OK);
        case BSIM1_CQBS:
            value->rValue = *(ckt->CKTstate0 + here->B1cqbs); 
            return(OK);
        case BSIM1_CDG:
            value->rValue = *(ckt->CKTstate0 + here->B1cdgb); 
            return(OK);
        case BSIM1_CDD:
            value->rValue = *(ckt->CKTstate0 + here->B1cddb); 
            return(OK);
        case BSIM1_CDS:
            value->rValue = *(ckt->CKTstate0 + here->B1cdsb); 
            return(OK);
        case BSIM1_VON:
            value->rValue = *(ckt->CKTstate0 + here->B1vono); 
            return(OK);
        case BSIM1_QBS:
            value->rValue = *(ckt->CKTstate0 + here->B1qbs); 
            return(OK);
        case BSIM1_QBD:
            value->rValue = *(ckt->CKTstate0 + here->B1qbd); 
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

