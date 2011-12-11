/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
Modified by Paolo Nenzi 2002
File: b3soifdask.c          98/5/01
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMFD2.1 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "b3soifddef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
B3SOIFDask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value,
           IFvalue *select)
{
B3SOIFDinstance *here = (B3SOIFDinstance*)inst;

    NG_IGNORE(select);

    switch(which) 
    {   case B3SOIFD_L:
            value->rValue = here->B3SOIFDl;
            return(OK);
        case B3SOIFD_W:
            value->rValue = here->B3SOIFDw;
            return(OK);
	case B3SOIFD_M:
            value->rValue = here->B3SOIFDm;
            return(OK);    
        case B3SOIFD_AS:
            value->rValue = here->B3SOIFDsourceArea;
            return(OK);
        case B3SOIFD_AD:
            value->rValue = here->B3SOIFDdrainArea;
            return(OK);
        case B3SOIFD_PS:
            value->rValue = here->B3SOIFDsourcePerimeter;
            return(OK);
        case B3SOIFD_PD:
            value->rValue = here->B3SOIFDdrainPerimeter;
            return(OK);
        case B3SOIFD_NRS:
            value->rValue = here->B3SOIFDsourceSquares;
            return(OK);
        case B3SOIFD_NRD:
            value->rValue = here->B3SOIFDdrainSquares;
            return(OK);
        case B3SOIFD_OFF:
            value->rValue = here->B3SOIFDoff;
            return(OK);
        case B3SOIFD_BJTOFF:
            value->iValue = here->B3SOIFDbjtoff;
            return(OK);
        case B3SOIFD_RTH0:
            value->rValue = here->B3SOIFDrth0;
	    value->rValue /= here->B3SOIFDm;
            return(OK);
        case B3SOIFD_CTH0:
            value->rValue = here->B3SOIFDcth0;
	    value->rValue *= here->B3SOIFDm;
            return(OK);
        case B3SOIFD_NRB:
            value->rValue = here->B3SOIFDbodySquares;
            return(OK);
        case B3SOIFD_IC_VBS:
            value->rValue = here->B3SOIFDicVBS;
            return(OK);
        case B3SOIFD_IC_VDS:
            value->rValue = here->B3SOIFDicVDS;
            return(OK);
        case B3SOIFD_IC_VGS:
            value->rValue = here->B3SOIFDicVGS;
            return(OK);
        case B3SOIFD_IC_VES:
            value->rValue = here->B3SOIFDicVES;
            return(OK);
        case B3SOIFD_IC_VPS:
            value->rValue = here->B3SOIFDicVPS;
            return(OK);
        case B3SOIFD_DNODE:
            value->iValue = here->B3SOIFDdNode;
            return(OK);
        case B3SOIFD_GNODE:
            value->iValue = here->B3SOIFDgNode;
            return(OK);
        case B3SOIFD_SNODE:
            value->iValue = here->B3SOIFDsNode;
            return(OK);
        case B3SOIFD_BNODE:
            value->iValue = here->B3SOIFDbNode;
            return(OK);
        case B3SOIFD_ENODE:
            value->iValue = here->B3SOIFDeNode;
            return(OK);
        case B3SOIFD_DNODEPRIME:
            value->iValue = here->B3SOIFDdNodePrime;
            return(OK);
        case B3SOIFD_SNODEPRIME:
            value->iValue = here->B3SOIFDsNodePrime;
            return(OK);
        case B3SOIFD_SOURCECONDUCT:
            value->rValue = here->B3SOIFDsourceConductance;
	    value->rValue *= here->B3SOIFDm;
            return(OK);
        case B3SOIFD_DRAINCONDUCT:
            value->rValue = here->B3SOIFDdrainConductance;
	    value->rValue *= here->B3SOIFDm;
            return(OK);
        case B3SOIFD_VBD:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIFDvbd);
            return(OK);
        case B3SOIFD_VBS:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIFDvbs);
            return(OK);
        case B3SOIFD_VGS:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIFDvgs);
            return(OK);
        case B3SOIFD_VES:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIFDves);
            return(OK);
        case B3SOIFD_VDS:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIFDvds);
            return(OK);
        case B3SOIFD_CD:
            value->rValue = here->B3SOIFDcd;
	    value->rValue *= here->B3SOIFDm; 
            return(OK);
        case B3SOIFD_CBS:
            value->rValue = here->B3SOIFDcjs;
	    value->rValue *= here->B3SOIFDm; 
            return(OK);
        case B3SOIFD_CBD:
            value->rValue = here->B3SOIFDcjd;
	    value->rValue *= here->B3SOIFDm; 
            return(OK);
        case B3SOIFD_GM:
            value->rValue = here->B3SOIFDgm; 
	    value->rValue *= here->B3SOIFDm;
            return(OK);
        case B3SOIFD_GMID:
            value->rValue = here->B3SOIFDgm/here->B3SOIFDcd; 
            return(OK);
        case B3SOIFD_GDS:
            value->rValue = here->B3SOIFDgds; 
	    value->rValue *= here->B3SOIFDm;
            return(OK);
        case B3SOIFD_GMBS:
            value->rValue = here->B3SOIFDgmbs;
	    value->rValue *= here->B3SOIFDm; 
            return(OK);
        case B3SOIFD_GBD:
            value->rValue = here->B3SOIFDgjdb; 
	    value->rValue *= here->B3SOIFDm;
            return(OK);
        case B3SOIFD_GBS:
            value->rValue = here->B3SOIFDgjsb;
	    value->rValue *= here->B3SOIFDm; 
            return(OK);
        case B3SOIFD_QB:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIFDqb); 
	    value->rValue *= here->B3SOIFDm;
            return(OK);
        case B3SOIFD_CQB:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIFDcqb);
	    value->rValue *= here->B3SOIFDm; 
            return(OK);
        case B3SOIFD_QG:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIFDqg); 
	    value->rValue *= here->B3SOIFDm;
            return(OK);
        case B3SOIFD_CQG:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIFDcqg);
	    value->rValue *= here->B3SOIFDm; 
            return(OK);
        case B3SOIFD_QD:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIFDqd); 
	    value->rValue *= here->B3SOIFDm;
            return(OK);
        case B3SOIFD_CQD:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIFDcqd); 
	    value->rValue *= here->B3SOIFDm;
            return(OK);
        case B3SOIFD_CGG:
            value->rValue = here->B3SOIFDcggb; 
	    value->rValue *= here->B3SOIFDm;
            return(OK);
        case B3SOIFD_CGD:
            value->rValue = here->B3SOIFDcgdb;
	    value->rValue *= here->B3SOIFDm;
            return(OK);
        case B3SOIFD_CGS:
            value->rValue = here->B3SOIFDcgsb;
	    value->rValue *= here->B3SOIFDm;
            return(OK);
        case B3SOIFD_CDG:
            value->rValue = here->B3SOIFDcdgb;
	    value->rValue *= here->B3SOIFDm; 
            return(OK);
        case B3SOIFD_CDD:
            value->rValue = here->B3SOIFDcddb;
	    value->rValue *= here->B3SOIFDm; 
            return(OK);
        case B3SOIFD_CDS:
            value->rValue = here->B3SOIFDcdsb; 
	    value->rValue *= here->B3SOIFDm;
            return(OK);
        case B3SOIFD_CBG:
            value->rValue = here->B3SOIFDcbgb;
	    value->rValue *= here->B3SOIFDm;
            return(OK);
        case B3SOIFD_CBDB:
            value->rValue = here->B3SOIFDcbdb;
	    value->rValue *= here->B3SOIFDm;
            return(OK);
        case B3SOIFD_CBSB:
            value->rValue = here->B3SOIFDcbsb;
	    value->rValue *= here->B3SOIFDm;
            return(OK);
        case B3SOIFD_VON:
            value->rValue = here->B3SOIFDvon; 
            return(OK);
        case B3SOIFD_VDSAT:
            value->rValue = here->B3SOIFDvdsat; 
            return(OK);
        case B3SOIFD_QBS:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIFDqbs);
	    value->rValue *= here->B3SOIFDm; 
            return(OK);
        case B3SOIFD_QBD:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIFDqbd); 
	    value->rValue *= here->B3SOIFDm;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

