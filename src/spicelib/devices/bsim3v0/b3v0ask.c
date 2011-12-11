/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
File: b3v0ask.c
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "bsim3v0def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BSIM3v0ask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value,
           IFvalue *select)
{
BSIM3v0instance *here = (BSIM3v0instance*)inst;

    NG_IGNORE(select);

    switch(which) 
    {   case BSIM3v0_L:
            value->rValue = here->BSIM3v0l;
            return(OK);
        case BSIM3v0_W:
            value->rValue = here->BSIM3v0w;
            return(OK);
	case BSIM3v0_M:
            value->rValue = here->BSIM3v0m;
            return(OK);   
        case BSIM3v0_AS:
            value->rValue = here->BSIM3v0sourceArea;
            return(OK);
        case BSIM3v0_AD:
            value->rValue = here->BSIM3v0drainArea;
            return(OK);
        case BSIM3v0_PS:
            value->rValue = here->BSIM3v0sourcePerimeter;
            return(OK);
        case BSIM3v0_PD:
            value->rValue = here->BSIM3v0drainPerimeter;
            return(OK);
        case BSIM3v0_NRS:
            value->rValue = here->BSIM3v0sourceSquares;
            return(OK);
        case BSIM3v0_NRD:
            value->rValue = here->BSIM3v0drainSquares;
            return(OK);
        case BSIM3v0_OFF:
            value->rValue = here->BSIM3v0off;
            return(OK);
        case BSIM3v0_NQSMOD:
            value->iValue = here->BSIM3v0nqsMod;
            return(OK);
        case BSIM3v0_IC_VBS:
            value->rValue = here->BSIM3v0icVBS;
            return(OK);
        case BSIM3v0_IC_VDS:
            value->rValue = here->BSIM3v0icVDS;
            return(OK);
        case BSIM3v0_IC_VGS:
            value->rValue = here->BSIM3v0icVGS;
            return(OK);
        case BSIM3v0_DNODE:
            value->iValue = here->BSIM3v0dNode;
            return(OK);
        case BSIM3v0_GNODE:
            value->iValue = here->BSIM3v0gNode;
            return(OK);
        case BSIM3v0_SNODE:
            value->iValue = here->BSIM3v0sNode;
            return(OK);
        case BSIM3v0_BNODE:
            value->iValue = here->BSIM3v0bNode;
            return(OK);
        case BSIM3v0_DNODEPRIME:
            value->iValue = here->BSIM3v0dNodePrime;
            return(OK);
        case BSIM3v0_SNODEPRIME:
            value->iValue = here->BSIM3v0sNodePrime;
            return(OK);
        case BSIM3v0_SOURCECONDUCT:
            value->rValue = here->BSIM3v0sourceConductance;
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        case BSIM3v0_DRAINCONDUCT:
            value->rValue = here->BSIM3v0drainConductance;
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        case BSIM3v0_VBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v0vbd);
            return(OK);
        case BSIM3v0_VBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v0vbs);
            return(OK);
        case BSIM3v0_VGS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v0vgs);
            return(OK);
        case BSIM3v0_VDS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v0vds);
            return(OK);
        case BSIM3v0_CD:
            value->rValue = here->BSIM3v0cd; 
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        case BSIM3v0_CBS:
            value->rValue = here->BSIM3v0cbs;
	    value->rValue *= here->BSIM3v0m; 
            return(OK);
        case BSIM3v0_CBD:
            value->rValue = here->BSIM3v0cbd; 
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        case BSIM3v0_GM:
            value->rValue = here->BSIM3v0gm; 
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        case BSIM3v0_GDS:
            value->rValue = here->BSIM3v0gds;
	    value->rValue *= here->BSIM3v0m; 
            return(OK);
        case BSIM3v0_GMBS:
            value->rValue = here->BSIM3v0gmbs;
	    value->rValue *= here->BSIM3v0m; 
            return(OK);
        case BSIM3v0_GBD:
            value->rValue = here->BSIM3v0gbd;
	    value->rValue *= here->BSIM3v0m; 
            return(OK);
        case BSIM3v0_GBS:
            value->rValue = here->BSIM3v0gbs;
	    value->rValue *= here->BSIM3v0m; 
            return(OK);
        case BSIM3v0_QB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v0qb);
	    value->rValue *= here->BSIM3v0m; 
            return(OK);
        case BSIM3v0_CQB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v0cqb); 
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        case BSIM3v0_QG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v0qg); 
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        case BSIM3v0_CQG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v0cqg);
	    value->rValue *= here->BSIM3v0m; 
            return(OK);
        case BSIM3v0_QD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v0qd); 
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        case BSIM3v0_CQD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v0cqd);
	    value->rValue *= here->BSIM3v0m; 
            return(OK);
        case BSIM3v0_CGG:
            value->rValue = here->BSIM3v0cggb; 
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        case BSIM3v0_CGD:
            value->rValue = here->BSIM3v0cgdb;
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        case BSIM3v0_CGS:
            value->rValue = here->BSIM3v0cgsb;
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        case BSIM3v0_CDG:
            value->rValue = here->BSIM3v0cdgb; 
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        case BSIM3v0_CDD:
            value->rValue = here->BSIM3v0cddb; 
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        case BSIM3v0_CDS:
            value->rValue = here->BSIM3v0cdsb; 
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        case BSIM3v0_CBG:
            value->rValue = here->BSIM3v0cbgb;
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        case BSIM3v0_CBDB:
            value->rValue = here->BSIM3v0cbdb;
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        case BSIM3v0_CBSB:
            value->rValue = here->BSIM3v0cbsb;
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        case BSIM3v0_CAPBD:
            value->rValue = here->BSIM3v0capbd; 
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        case BSIM3v0_CAPBS:
            value->rValue = here->BSIM3v0capbs;
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        case BSIM3v0_VON:
            value->rValue = here->BSIM3v0von; 
            return(OK);
        case BSIM3v0_VDSAT:
            value->rValue = here->BSIM3v0vdsat; 
            return(OK);
        case BSIM3v0_QBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v0qbs); 
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        case BSIM3v0_QBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v0qbd); 
	    value->rValue *= here->BSIM3v0m;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

