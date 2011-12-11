/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipdask.c          98/5/01
Modified by Pin Su	99/4/30
Modified by Pin Su      01/2/15
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.2.3  02/3/5  Pin Su 
 * BSIMPD2.2.3 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "b3soipddef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
B3SOIPDask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value,
           IFvalue *select)
{
B3SOIPDinstance *here = (B3SOIPDinstance*)inst;

    NG_IGNORE(select);

    switch(which) 
    {   case B3SOIPD_L:
            value->rValue = here->B3SOIPDl;
            return(OK);
        case B3SOIPD_W:
            value->rValue = here->B3SOIPDw;
            return(OK);
	case B3SOIPD_M:
            value->rValue = here->B3SOIPDm;
            return(OK);
        case B3SOIPD_AS:
            value->rValue = here->B3SOIPDsourceArea;
            return(OK);
        case B3SOIPD_AD:
            value->rValue = here->B3SOIPDdrainArea;
            return(OK);
        case B3SOIPD_PS:
            value->rValue = here->B3SOIPDsourcePerimeter;
            return(OK);
        case B3SOIPD_PD:
            value->rValue = here->B3SOIPDdrainPerimeter;
            return(OK);
        case B3SOIPD_NRS:
            value->rValue = here->B3SOIPDsourceSquares;
            return(OK);
        case B3SOIPD_NRD:
            value->rValue = here->B3SOIPDdrainSquares;
            return(OK);
        case B3SOIPD_OFF:
            value->iValue = here->B3SOIPDoff;
            return(OK);
        case B3SOIPD_BJTOFF:
            value->iValue = here->B3SOIPDbjtoff;
            return(OK);
        case B3SOIPD_RTH0:
            value->rValue = here->B3SOIPDrth0;
	    value->rValue /= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_CTH0:
            value->rValue = here->B3SOIPDcth0;
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_NRB:
            value->rValue = here->B3SOIPDbodySquares;
            return(OK);
        case B3SOIPD_FRBODY:
            value->rValue = here->B3SOIPDfrbody;
            return(OK);


/* v2.0 release */
        case B3SOIPD_NBC:
            value->rValue = here->B3SOIPDnbc;
            return(OK);
        case B3SOIPD_NSEG:
            value->rValue = here->B3SOIPDnseg;
            return(OK);
        case B3SOIPD_PDBCP:
            value->rValue = here->B3SOIPDpdbcp;
            return(OK);
        case B3SOIPD_PSBCP:
            value->rValue = here->B3SOIPDpsbcp;
            return(OK);
        case B3SOIPD_AGBCP:
            value->rValue = here->B3SOIPDagbcp;
            return(OK);
        case B3SOIPD_AEBCP:
            value->rValue = here->B3SOIPDaebcp;
            return(OK);
        case B3SOIPD_VBSUSR:
            value->rValue = here->B3SOIPDvbsusr;
            return(OK);
        case B3SOIPD_TNODEOUT:
            value->iValue = here->B3SOIPDtnodeout;
            return(OK);


        case B3SOIPD_IC_VBS:
            value->rValue = here->B3SOIPDicVBS;
            return(OK);
        case B3SOIPD_IC_VDS:
            value->rValue = here->B3SOIPDicVDS;
            return(OK);
        case B3SOIPD_IC_VGS:
            value->rValue = here->B3SOIPDicVGS;
            return(OK);
        case B3SOIPD_IC_VES:
            value->rValue = here->B3SOIPDicVES;
            return(OK);
        case B3SOIPD_IC_VPS:
            value->rValue = here->B3SOIPDicVPS;
            return(OK);
        case B3SOIPD_DNODE:
            value->iValue = here->B3SOIPDdNode;
            return(OK);
        case B3SOIPD_GNODE:
            value->iValue = here->B3SOIPDgNode;
            return(OK);
        case B3SOIPD_SNODE:
            value->iValue = here->B3SOIPDsNode;
            return(OK);
        case B3SOIPD_BNODE:
            value->iValue = here->B3SOIPDbNode;
            return(OK);
        case B3SOIPD_ENODE:
            value->iValue = here->B3SOIPDeNode;
            return(OK);
        case B3SOIPD_DNODEPRIME:
            value->iValue = here->B3SOIPDdNodePrime;
            return(OK);
        case B3SOIPD_SNODEPRIME:
            value->iValue = here->B3SOIPDsNodePrime;
            return(OK);
        case B3SOIPD_SOURCECONDUCT:
            value->rValue = here->B3SOIPDsourceConductance;
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_DRAINCONDUCT:
            value->rValue = here->B3SOIPDdrainConductance;
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_VBD:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIPDvbd);
            return(OK);
        case B3SOIPD_VBS:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIPDvbs);
            return(OK);
        case B3SOIPD_VGS:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIPDvgs);
            return(OK);
        case B3SOIPD_VES:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIPDves);
            return(OK);
        case B3SOIPD_VDS:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIPDvds);
            return(OK);
        case B3SOIPD_CD:
            value->rValue = here->B3SOIPDcd; 
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_CBS:
            value->rValue = here->B3SOIPDcjs; 
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_CBD:
            value->rValue = here->B3SOIPDcjd; 
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_GM:
            value->rValue = here->B3SOIPDgm; 
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_GMID:
            value->rValue = here->B3SOIPDgm/here->B3SOIPDcd; 
            return(OK);
        case B3SOIPD_GDS:
            value->rValue = here->B3SOIPDgds;
	    value->rValue *= here->B3SOIPDm; 
            return(OK);
        case B3SOIPD_GMBS:
            value->rValue = here->B3SOIPDgmbs;
	    value->rValue *= here->B3SOIPDm; 
            return(OK);
        case B3SOIPD_GBD:
            value->rValue = here->B3SOIPDgjdb; 
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_GBS:
            value->rValue = here->B3SOIPDgjsb; 
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_QB:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIPDqb);
	    value->rValue *= here->B3SOIPDm; 
            return(OK);
        case B3SOIPD_CQB:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIPDcqb);
	    value->rValue *= here->B3SOIPDm; 
            return(OK);
        case B3SOIPD_QG:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIPDqg); 
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_CQG:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIPDcqg); 
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_QD:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIPDqd);
	    value->rValue *= here->B3SOIPDm; 
            return(OK);
        case B3SOIPD_CQD:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIPDcqd); 
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_CGG:
            value->rValue = here->B3SOIPDcggb; 
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_CGD:
            value->rValue = here->B3SOIPDcgdb;
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_CGS:
            value->rValue = here->B3SOIPDcgsb;
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_CDG:
            value->rValue = here->B3SOIPDcdgb; 
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_CDD:
            value->rValue = here->B3SOIPDcddb; 
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_CDS:
            value->rValue = here->B3SOIPDcdsb;
	    value->rValue *= here->B3SOIPDm; 
            return(OK);
        case B3SOIPD_CBG:
            value->rValue = here->B3SOIPDcbgb;
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_CBDB:
            value->rValue = here->B3SOIPDcbdb;
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_CBSB:
            value->rValue = here->B3SOIPDcbsb;
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_VON:
            value->rValue = here->B3SOIPDvon; 
            return(OK);
        case B3SOIPD_VDSAT:
            value->rValue = here->B3SOIPDvdsat; 
            return(OK);
        case B3SOIPD_QBS:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIPDqbs); 
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        case B3SOIPD_QBD:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIPDqbd); 
	    value->rValue *= here->B3SOIPDm;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

