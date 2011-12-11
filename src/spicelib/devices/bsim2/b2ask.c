/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Hong J. Park
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "bsim2def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*ARGSUSED*/
int
B2ask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value,
      IFvalue *select)
{
    B2instance *here = (B2instance*)inst;

    NG_IGNORE(select);

    switch(which) {
        case BSIM2_L:
            value->rValue = here->B2l;
            return(OK);
        case BSIM2_W:
            value->rValue = here->B2w;
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_M:
            value->rValue = here->B2m;
            return(OK);
        case BSIM2_AS:
            value->rValue = here->B2sourceArea;
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_AD:
            value->rValue = here->B2drainArea;
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_PS:
            value->rValue = here->B2sourcePerimeter;
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_PD:
            value->rValue = here->B2drainPerimeter;
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_NRS:
            value->rValue = here->B2sourceSquares;
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_NRD:
            value->rValue = here->B2drainSquares;
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_OFF:
            value->rValue = here->B2off;
            return(OK);
        case BSIM2_IC_VBS:
            value->rValue = here->B2icVBS;
            return(OK);
        case BSIM2_IC_VDS:
            value->rValue = here->B2icVDS;
            return(OK);
        case BSIM2_IC_VGS:
            value->rValue = here->B2icVGS;
            return(OK);
        case BSIM2_DNODE:
            value->iValue = here->B2dNode;
            return(OK);
        case BSIM2_GNODE:
            value->iValue = here->B2gNode;
            return(OK);
        case BSIM2_SNODE:
            value->iValue = here->B2sNode;
            return(OK);
        case BSIM2_BNODE:
            value->iValue = here->B2bNode;
            return(OK);
        case BSIM2_DNODEPRIME:
            value->iValue = here->B2dNodePrime;
            return(OK);
        case BSIM2_SNODEPRIME:
            value->iValue = here->B2sNodePrime;
            return(OK);
        case BSIM2_SOURCECONDUCT:
            value->rValue = here->B2sourceConductance;
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_DRAINCONDUCT:
            value->rValue = here->B2drainConductance;
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_VBD:
            value->rValue = *(ckt->CKTstate0 + here->B2vbd);
            return(OK);
        case BSIM2_VBS:
            value->rValue = *(ckt->CKTstate0 + here->B2vbs);
            return(OK);
        case BSIM2_VGS:
            value->rValue = *(ckt->CKTstate0 + here->B2vgs);
            return(OK);
        case BSIM2_VDS:
            value->rValue = *(ckt->CKTstate0 + here->B2vds);
            return(OK);
        case BSIM2_CD:
            value->rValue = *(ckt->CKTstate0 + here->B2cd); 
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_CBS:
            value->rValue = *(ckt->CKTstate0 + here->B2cbs);
            value->rValue *= here->B2m; 
            return(OK);
        case BSIM2_CBD:
            value->rValue = *(ckt->CKTstate0 + here->B2cbd);
            value->rValue *= here->B2m; 
            return(OK);
        case BSIM2_GM:
            value->rValue = *(ckt->CKTstate0 + here->B2gm); 
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_GDS:
            value->rValue = *(ckt->CKTstate0 + here->B2gds);
            value->rValue *= here->B2m; 
            return(OK);
        case BSIM2_GMBS:
            value->rValue = *(ckt->CKTstate0 + here->B2gmbs); 
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_GBD:
            value->rValue = *(ckt->CKTstate0 + here->B2gbd); 
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_GBS:
            value->rValue = *(ckt->CKTstate0 + here->B2gbs); 
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_QB:
            value->rValue = *(ckt->CKTstate0 + here->B2qb); 
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_CQB:
            value->rValue = *(ckt->CKTstate0 + here->B2cqb); 
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_QG:
            value->rValue = *(ckt->CKTstate0 + here->B2qg); 
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_CQG:
            value->rValue = *(ckt->CKTstate0 + here->B2cqg); 
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_QD:
            value->rValue = *(ckt->CKTstate0 + here->B2qd); 
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_CQD:
            value->rValue = *(ckt->CKTstate0 + here->B2cqd); 
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_CGG:
            value->rValue = *(ckt->CKTstate0 + here->B2cggb); 
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_CGD:
            value->rValue = *(ckt->CKTstate0 + here->B2cgdb); 
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_CGS:
            value->rValue = *(ckt->CKTstate0 + here->B2cgsb); 
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_CBG:
            value->rValue = *(ckt->CKTstate0 + here->B2cbgb); 
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_CAPBD:
            value->rValue = *(ckt->CKTstate0 + here->B2capbd);
            value->rValue *= here->B2m; 
            return(OK);
        case BSIM2_CQBD:
            value->rValue = *(ckt->CKTstate0 + here->B2cqbd); 
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_CAPBS:
            value->rValue = *(ckt->CKTstate0 + here->B2capbs); 
            value->rValue *= here->B2m;            
            return(OK);
        case BSIM2_CQBS:
            value->rValue = *(ckt->CKTstate0 + here->B2cqbs); 
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_CDG:
            value->rValue = *(ckt->CKTstate0 + here->B2cdgb);
            value->rValue *= here->B2m; 
            return(OK);
        case BSIM2_CDD:
            value->rValue = *(ckt->CKTstate0 + here->B2cddb);
            value->rValue *= here->B2m; 
            return(OK);
        case BSIM2_CDS:
            value->rValue = *(ckt->CKTstate0 + here->B2cdsb); 
            value->rValue *= here->B2m;
            return(OK);
        case BSIM2_VON:
            value->rValue = *(ckt->CKTstate0 + here->B2vono); 
            return(OK);
        case BSIM2_QBS:
            value->rValue = *(ckt->CKTstate0 + here->B2qbs);
            value->rValue *= here->B2m; 
            return(OK);
        case BSIM2_QBD:
            value->rValue = *(ckt->CKTstate0 + here->B2qbd); 
            value->rValue *= here->B2m;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}


