/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: Weidong Liu and Pin Su         Feb 1999
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soiddask.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMDD2.1 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "b3soidddef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
B3SOIDDask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value,
           IFvalue *select)
{
B3SOIDDinstance *here = (B3SOIDDinstance*)inst;

    NG_IGNORE(select);

    switch(which) 
    {   case B3SOIDD_L:
            value->rValue = here->B3SOIDDl;
            return(OK);
        case B3SOIDD_W:
            value->rValue = here->B3SOIDDw;
            return(OK);
        case B3SOIDD_AS:
            value->rValue = here->B3SOIDDsourceArea;
            return(OK);
        case B3SOIDD_AD:
            value->rValue = here->B3SOIDDdrainArea;
            return(OK);
        case B3SOIDD_PS:
            value->rValue = here->B3SOIDDsourcePerimeter;
            return(OK);
        case B3SOIDD_PD:
            value->rValue = here->B3SOIDDdrainPerimeter;
            return(OK);
        case B3SOIDD_NRS:
            value->rValue = here->B3SOIDDsourceSquares;
            return(OK);
        case B3SOIDD_NRD:
            value->rValue = here->B3SOIDDdrainSquares;
            return(OK);
        case B3SOIDD_OFF:
            value->rValue = here->B3SOIDDoff;
            return(OK);
        case B3SOIDD_BJTOFF:
            value->iValue = here->B3SOIDDbjtoff;
            return(OK);
        case B3SOIDD_RTH0:
            value->rValue = here->B3SOIDDrth0;
	    value->rValue /= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_CTH0:
            value->rValue = here->B3SOIDDcth0;
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_NRB:
            value->rValue = here->B3SOIDDbodySquares;
            return(OK);
        case B3SOIDD_IC_VBS:
            value->rValue = here->B3SOIDDicVBS;
            return(OK);
        case B3SOIDD_IC_VDS:
            value->rValue = here->B3SOIDDicVDS;
            return(OK);
        case B3SOIDD_IC_VGS:
            value->rValue = here->B3SOIDDicVGS;
            return(OK);
        case B3SOIDD_IC_VES:
            value->rValue = here->B3SOIDDicVES;
            return(OK);
        case B3SOIDD_IC_VPS:
            value->rValue = here->B3SOIDDicVPS;
            return(OK);
        case B3SOIDD_DNODE:
            value->iValue = here->B3SOIDDdNode;
            return(OK);
        case B3SOIDD_GNODE:
            value->iValue = here->B3SOIDDgNode;
            return(OK);
        case B3SOIDD_SNODE:
            value->iValue = here->B3SOIDDsNode;
            return(OK);
        case B3SOIDD_BNODE:
            value->iValue = here->B3SOIDDbNode;
            return(OK);
        case B3SOIDD_ENODE:
            value->iValue = here->B3SOIDDeNode;
            return(OK);
        case B3SOIDD_DNODEPRIME:
            value->iValue = here->B3SOIDDdNodePrime;
            return(OK);
        case B3SOIDD_SNODEPRIME:
            value->iValue = here->B3SOIDDsNodePrime;
            return(OK);
        case B3SOIDD_SOURCECONDUCT:
            value->rValue = here->B3SOIDDsourceConductance;
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_DRAINCONDUCT:
            value->rValue = here->B3SOIDDdrainConductance;
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_VBD:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIDDvbd);
            return(OK);
        case B3SOIDD_VBS:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIDDvbs);
            return(OK);
        case B3SOIDD_VGS:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIDDvgs);
            return(OK);
        case B3SOIDD_VES:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIDDves);
            return(OK);
        case B3SOIDD_VDS:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIDDvds);
            return(OK);
        case B3SOIDD_CD:
            value->rValue = here->B3SOIDDcd; 
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_CBS:
            value->rValue = here->B3SOIDDcjs; 
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_CBD:
            value->rValue = here->B3SOIDDcjd;
	    value->rValue *= here->B3SOIDDm; 
            return(OK);
        case B3SOIDD_GM:
            value->rValue = here->B3SOIDDgm; 
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_GMID:
            value->rValue = here->B3SOIDDgm/here->B3SOIDDcd; 
            return(OK);
        case B3SOIDD_GDS:
            value->rValue = here->B3SOIDDgds; 
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_GMBS:
            value->rValue = here->B3SOIDDgmbs; 
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_GBD:
            value->rValue = here->B3SOIDDgjdb; 
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_GBS:
            value->rValue = here->B3SOIDDgjsb; 
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_QB:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIDDqb); 
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_CQB:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIDDcqb); 
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_QG:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIDDqg); 
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_CQG:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIDDcqg);
	    value->rValue *= here->B3SOIDDm; 
            return(OK);
        case B3SOIDD_QD:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIDDqd); 
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_CQD:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIDDcqd); 
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_CGG:
            value->rValue = here->B3SOIDDcggb;
	    value->rValue *= here->B3SOIDDm; 
            return(OK);
        case B3SOIDD_CGD:
            value->rValue = here->B3SOIDDcgdb;
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_CGS:
            value->rValue = here->B3SOIDDcgsb;
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_CDG:
            value->rValue = here->B3SOIDDcdgb; 
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_CDD:
            value->rValue = here->B3SOIDDcddb; 
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_CDS:
            value->rValue = here->B3SOIDDcdsb; 
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_CBG:
            value->rValue = here->B3SOIDDcbgb;
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_CBDB:
            value->rValue = here->B3SOIDDcbdb;
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_CBSB:
            value->rValue = here->B3SOIDDcbsb;
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_VON:
            value->rValue = here->B3SOIDDvon; 
            return(OK);
        case B3SOIDD_VDSAT:
            value->rValue = here->B3SOIDDvdsat; 
            return(OK);
        case B3SOIDD_QBS:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIDDqbs); 
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        case B3SOIDD_QBD:
            value->rValue = *(ckt->CKTstate0 + here->B3SOIDDqbd); 
	    value->rValue *= here->B3SOIDDm;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

