/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1995 Min-Chie Jeng and Mansun Chan.
Modified by Paolo Nenzi 2002
File: b3v1aask.c
**********/

#include "ngspice.h"
#include "ifsim.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "bsim3v1adef.h"
#include "sperror.h"
#include "suffix.h"

int
BSIM3v1Aask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value,
            IFvalue *select)

{
BSIM3v1Ainstance *here = (BSIM3v1Ainstance*)inst;

    switch(which) 
    {   case BSIM3v1A_L:
            value->rValue = here->BSIM3v1Al;
            return(OK);
        case BSIM3v1A_W:
            value->rValue = here->BSIM3v1Aw;
            return(OK);
        case BSIM3v1A_M:
            value->rValue = here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_AS:
            value->rValue = here->BSIM3v1AsourceArea;
            return(OK);
        case BSIM3v1A_AD:
            value->rValue = here->BSIM3v1AdrainArea;
            return(OK);
        case BSIM3v1A_PS:
            value->rValue = here->BSIM3v1AsourcePerimeter;
            return(OK);
        case BSIM3v1A_PD:
            value->rValue = here->BSIM3v1AdrainPerimeter;
            return(OK);
        case BSIM3v1A_NRS:
            value->rValue = here->BSIM3v1AsourceSquares;
            return(OK);
        case BSIM3v1A_NRD:
            value->rValue = here->BSIM3v1AdrainSquares;
            return(OK);
        case BSIM3v1A_OFF:
            value->rValue = here->BSIM3v1Aoff;
            return(OK);
        case BSIM3v1A_NQSMOD:
            value->iValue = here->BSIM3v1AnqsMod;
            return(OK);
        case BSIM3v1A_IC_VBS:
            value->rValue = here->BSIM3v1AicVBS;
            return(OK);
        case BSIM3v1A_IC_VDS:
            value->rValue = here->BSIM3v1AicVDS;
            return(OK);
        case BSIM3v1A_IC_VGS:
            value->rValue = here->BSIM3v1AicVGS;
            return(OK);
        case BSIM3v1A_DNODE:
            value->iValue = here->BSIM3v1AdNode;
            return(OK);
        case BSIM3v1A_GNODE:
            value->iValue = here->BSIM3v1AgNode;
            return(OK);
        case BSIM3v1A_SNODE:
            value->iValue = here->BSIM3v1AsNode;
            return(OK);
        case BSIM3v1A_BNODE:
            value->iValue = here->BSIM3v1AbNode;
            return(OK);
        case BSIM3v1A_DNODEPRIME:
            value->iValue = here->BSIM3v1AdNodePrime;
            return(OK);
        case BSIM3v1A_SNODEPRIME:
            value->iValue = here->BSIM3v1AsNodePrime;
            return(OK);
        case BSIM3v1A_SOURCECONDUCT:
            value->rValue = here->BSIM3v1AsourceConductance;
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_DRAINCONDUCT:
            value->rValue = here->BSIM3v1AdrainConductance;
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_VBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Avbd);
            return(OK);
        case BSIM3v1A_VBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Avbs);
            return(OK);
        case BSIM3v1A_VGS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Avgs);
            return(OK);
        case BSIM3v1A_VDS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Avds);
            return(OK);
        case BSIM3v1A_CD:
            value->rValue = here->BSIM3v1Acd; 
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_CBS:
            value->rValue = here->BSIM3v1Acbs; 
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_CBD:
            value->rValue = here->BSIM3v1Acbd; 
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_GM:
            value->rValue = here->BSIM3v1Agm; 
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_GDS:
            value->rValue = here->BSIM3v1Agds; 
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_GMBS:
            value->rValue = here->BSIM3v1Agmbs; 
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_GBD:
            value->rValue = here->BSIM3v1Agbd; 
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_GBS:
            value->rValue = here->BSIM3v1Agbs; 
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_QB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Aqb); 
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_CQB:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Acqb); 
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_QG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Aqg);  
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_CQG:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Acqg); 
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_QD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Aqd); 
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_CQD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Acqd);  
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_CGG:
            value->rValue = here->BSIM3v1Acggb; 
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_CGD:
            value->rValue = here->BSIM3v1Acgdb;
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_CGS:
            value->rValue = here->BSIM3v1Acgsb;
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_CDG:
            value->rValue = here->BSIM3v1Acdgb; 
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_CDD:
            value->rValue = here->BSIM3v1Acddb; 
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_CDS:
            value->rValue = here->BSIM3v1Acdsb; 
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_CBG:
            value->rValue = here->BSIM3v1Acbgb;
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_CBDB:
            value->rValue = here->BSIM3v1Acbdb;
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_CBSB:
            value->rValue = here->BSIM3v1Acbsb;
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_CAPBD:
            value->rValue = here->BSIM3v1Acapbd; 
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_CAPBS:
            value->rValue = here->BSIM3v1Acapbs;
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_VON:
            value->rValue = here->BSIM3v1Avon; 
            return(OK);
        case BSIM3v1A_VDSAT:
            value->rValue = here->BSIM3v1Avdsat; 
            return(OK);
        case BSIM3v1A_QBS:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Aqbs); 
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        case BSIM3v1A_QBD:
            value->rValue = *(ckt->CKTstate0 + here->BSIM3v1Aqbd);  
            value->rValue *= here->BSIM3v1Am;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

