/***  B4SOI 12/16/2010 Released by Tanvir Morshed   ***/


/**********
 * Copyright 2010 Regents of the University of California.  All rights reserved.
 * Authors: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
 * Authors: 1999-2004 Pin Su, Hui Wan, Wei Jin, b3soiask.c
 * Authors: 2005- Hui Wan, Xuemei Xi, Ali Niknejad, Chenming Hu.
 * Authors: 2009- Wenwei Yang, Chung-Hsun Lin, Ali Niknejad, Chenming Hu.
 * File: b4soiask.c
 * Modified by Hui Wan, Xuemei Xi 11/30/2005
 * Modified by Wenwei Yang, Chung-Hsun Lin, Darsen Lu 03/06/2009
 * Modified by Tanvir Morshed 09/22/2009
 * Modified by Tanvir Morshed 12/31/2009
 **********/

#include "ngspice/ngspice.h"

#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "b4soidef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
B4SOIask(
CKTcircuit *ckt,
GENinstance *inst,
int which,
IFvalue *value,
IFvalue *select)
{
B4SOIinstance *here = (B4SOIinstance*)inst;

    NG_IGNORE(select);

    switch(which) 
    {   case B4SOI_L:
            value->rValue = here->B4SOIl;
            return(OK);
        case B4SOI_W:
            value->rValue = here->B4SOIw;
            return(OK);
        case B4SOI_M:
            value->rValue = here->B4SOIm;
            return(OK);
        case B4SOI_AS:
            value->rValue = here->B4SOIsourceArea;
            return(OK);
        case B4SOI_AD:
            value->rValue = here->B4SOIdrainArea;
            return(OK);
        case B4SOI_PS:
            value->rValue = here->B4SOIsourcePerimeter;
            return(OK);
        case B4SOI_PD:
            value->rValue = here->B4SOIdrainPerimeter;
            return(OK);
        case B4SOI_NRS:
            value->rValue = here->B4SOIsourceSquares;
            return(OK);
        case B4SOI_NRD:
            value->rValue = here->B4SOIdrainSquares;
            return(OK);
        case B4SOI_OFF:
            value->iValue = here->B4SOIoff;
            return(OK);
        case B4SOI_BJTOFF:
            value->iValue = here->B4SOIbjtoff;
            return(OK);
        case B4SOI_RTH0:
            value->rValue = here->B4SOIrth0;
            return(OK);
        case B4SOI_CTH0:
            value->rValue = here->B4SOIcth0;
            return(OK);
        case B4SOI_NRB:
            value->rValue = here->B4SOIbodySquares;
            return(OK);
        case B4SOI_FRBODY:
            value->rValue = here->B4SOIfrbody;
            return(OK);

        case B4SOI_QB:
            value->rValue = here->B4SOIqbulk;
            return(OK);
        case B4SOI_QD:
            value->rValue = here->B4SOIqdrn;
            return(OK);
        case B4SOI_QS:
            value->rValue = here->B4SOIqsrc;
            return(OK);
        case B4SOI_CGG:
            value->rValue = here->B4SOIcggb;
            return(OK);
        case B4SOI_CGD:
            value->rValue = here->B4SOIcgdb;
            return(OK);
        case B4SOI_CGS:
            value->rValue = here->B4SOIcgsb;
            return(OK);
        case B4SOI_CDG:
            value->rValue = here->B4SOIcdgb;
            return(OK);
        case B4SOI_CDD:
            value->rValue = here->B4SOIcddb;
            return(OK);
        case B4SOI_CDS:
            value->rValue = here->B4SOIcdsb;
            return(OK);
        case B4SOI_CBG:
            value->rValue = here->B4SOIcbgb;
            return(OK);
        case B4SOI_CBD:
            value->rValue = here->B4SOIcbdb;
            return(OK);
        case B4SOI_CBS:
            value->rValue = here->B4SOIcbsb;
            return(OK);
        case B4SOI_CAPBD:
            value->rValue = here->B4SOIcapbd;
            return(OK);
        case B4SOI_CAPBS:
            value->rValue = here->B4SOIcapbs;
            return(OK);

/* v4.0 */
        case B4SOI_RBSB:
            value->rValue = here->B4SOIrbsb;
            return(OK);
        case B4SOI_RBDB:
            value->rValue = here->B4SOIrbdb;
            return(OK);
        case B4SOI_CJSB:
            value->rValue = here->B4SOIcjsb;
            return(OK);
        case B4SOI_CJDB:
            value->rValue = here->B4SOIcjdb;
            return(OK);
        case B4SOI_SA:
            value->rValue = here->B4SOIsa ;
            return(OK);
        case B4SOI_SB:
            value->rValue = here->B4SOIsb ;
            return(OK);
        case B4SOI_SD:
            value->rValue = here->B4SOIsd ;
            return(OK);
        case B4SOI_RBODYMOD:
            value->iValue = here->B4SOIrbodyMod;
            return(OK);
        case B4SOI_NF:
            value->rValue = here->B4SOInf;
            return(OK);
        case B4SOI_DELVTO:
            value->rValue = here->B4SOIdelvto;
            return(OK);

/* v4.0 end */

/* v3.2 */
        case B4SOI_SOIMOD:
            value->iValue = here->B4SOIsoiMod;
            return(OK);

/* v3.1 added rgate */
        case B4SOI_RGATEMOD:
            value->iValue = here->B4SOIrgateMod;
            return(OK);
/* v3.1 added rgate end */

/* v2.0 release */
        case B4SOI_NBC:
            value->rValue = here->B4SOInbc;
            return(OK);
        case B4SOI_NSEG:
            value->rValue = here->B4SOInseg;
            return(OK);
        case B4SOI_PDBCP:
            value->rValue = here->B4SOIpdbcp;
            return(OK);
        case B4SOI_PSBCP:
            value->rValue = here->B4SOIpsbcp;
            return(OK);
        case B4SOI_AGBCP:
            value->rValue = here->B4SOIagbcp;
            return(OK);
     case B4SOI_AGBCP2:
            value->rValue = here->B4SOIagbcp2;
            return(OK);         /* v4.1 for BC improvement */
        case B4SOI_AGBCPD:        /* v4.0 */
            value->rValue = here->B4SOIagbcpd;
            return(OK);
        case B4SOI_AEBCP:
            value->rValue = here->B4SOIaebcp;
            return(OK);
        case B4SOI_VBSUSR:
            value->rValue = here->B4SOIvbsusr;
            return(OK);
        case B4SOI_TNODEOUT:
            value->iValue = here->B4SOItnodeout;
            return(OK);


        case B4SOI_IC_VBS:
            value->rValue = here->B4SOIicVBS;
            return(OK);
        case B4SOI_IC_VDS:
            value->rValue = here->B4SOIicVDS;
            return(OK);
        case B4SOI_IC_VGS:
            value->rValue = here->B4SOIicVGS;
            return(OK);
        case B4SOI_IC_VES:
            value->rValue = here->B4SOIicVES;
            return(OK);
        case B4SOI_IC_VPS:
            value->rValue = here->B4SOIicVPS;
            return(OK);
        case B4SOI_DNODE:
            value->iValue = here->B4SOIdNode;
            return(OK);
        case B4SOI_GNODE:
            value->iValue = here->B4SOIgNode;
            return(OK);
        case B4SOI_SNODE:
            value->iValue = here->B4SOIsNode;
            return(OK);
        case B4SOI_BNODE:
            value->iValue = here->B4SOIbNode;
            return(OK);
        case B4SOI_ENODE:
            value->iValue = here->B4SOIeNode;
            return(OK);
        case B4SOI_DNODEPRIME:
            value->iValue = here->B4SOIdNodePrime;
            return(OK);
        case B4SOI_SNODEPRIME:
            value->iValue = here->B4SOIsNodePrime;
            return(OK);

/* v3.1 added for RF */
        case B4SOI_GNODEEXT:
            value->iValue = here->B4SOIgNodeExt;
            return(OK);
        case B4SOI_GNODEMID:
            value->iValue = here->B4SOIgNodeMid;
            return(OK);
/* added for RF end*/


        case B4SOI_SOURCECONDUCT:
            value->rValue = here->B4SOIsourceConductance;
            value->rValue *= here->B4SOIm;
            return(OK);
        case B4SOI_DRAINCONDUCT:
            value->rValue = here->B4SOIdrainConductance;
            value->rValue *= here->B4SOIm;
            return(OK);
        case B4SOI_VBD:
            value->rValue = *(ckt->CKTstate0 + here->B4SOIvbd);
            return(OK);
        case B4SOI_VBS:
            value->rValue = *(ckt->CKTstate0 + here->B4SOIvbs);
            return(OK);
        case B4SOI_VGS:
            value->rValue = *(ckt->CKTstate0 + here->B4SOIvgs);
            return(OK);
        case B4SOI_VES:
            value->rValue = *(ckt->CKTstate0 + here->B4SOIves);
            return(OK);
        case B4SOI_VDS:
            value->rValue = *(ckt->CKTstate0 + here->B4SOIvds);
            return(OK);
        case B4SOI_CD:
            value->rValue = here->B4SOIcdrain; 
            value->rValue *= here->B4SOIm;
            return(OK);
        case B4SOI_IBS:
            value->rValue = here->B4SOIibs;
            value->rValue *= here->B4SOIm;
            return(OK);
        case B4SOI_IBD:
            value->rValue = here->B4SOIibd;
            value->rValue *= here->B4SOIm;
            return(OK);
        case B4SOI_ISUB:
            value->rValue = here->B4SOIiii;
            value->rValue *= here->B4SOIm;
            return(OK);
        case B4SOI_IGIDL:
            value->rValue = here->B4SOIigidl;
            return(OK);
                case B4SOI_IGISL:
            value->rValue = here->B4SOIigisl;
            return(OK);
        case B4SOI_IGS:
            value->rValue = here->B4SOIIgs;
            value->rValue *= here->B4SOIm;
            return(OK);
        case B4SOI_IGD:
            value->rValue = here->B4SOIIgd;
            value->rValue *= here->B4SOIm;
            return(OK);
        case B4SOI_IGB:
            value->rValue = here->B4SOIIgb;
            value->rValue *= here->B4SOIm;
            return(OK);
        case B4SOI_IGCS:
            value->rValue = here->B4SOIIgcs;
            value->rValue *= here->B4SOIm;
            return(OK);
        case B4SOI_IGCD:
            value->rValue = here->B4SOIIgcd;
            value->rValue *= here->B4SOIm;
            return(OK);
        case B4SOI_GM:
            value->rValue = here->B4SOIgm; 
            value->rValue *= here->B4SOIm;
            return(OK);
        case B4SOI_GMID:
            value->rValue = here->B4SOIgm/here->B4SOIcd; 
            return(OK);
        case B4SOI_GDS:
            value->rValue = here->B4SOIgds; 
            value->rValue *= here->B4SOIm;
            return(OK);
        case B4SOI_GMBS:
            value->rValue = here->B4SOIgmbs; 
            value->rValue *= here->B4SOIm;
            return(OK);
        case B4SOI_GBD:
            value->rValue = here->B4SOIgjdb; 
            value->rValue *= here->B4SOIm;
            return(OK);
        case B4SOI_GBS:
            value->rValue = here->B4SOIgjsb; 
            value->rValue *= here->B4SOIm;
            return(OK);
        case B4SOI_CQB:
            value->rValue = *(ckt->CKTstate0 + here->B4SOIcqb); 
            return(OK);
        case B4SOI_CQG:
            value->rValue = *(ckt->CKTstate0 + here->B4SOIcqg); 
            return(OK);
        case B4SOI_CQD:
            value->rValue = *(ckt->CKTstate0 + here->B4SOIcqd); 
            return(OK);
        case B4SOI_CBDB:
            value->rValue = here->B4SOIcbdb;
            value->rValue *= here->B4SOIm;
            return(OK);
        case B4SOI_CBSB:
            value->rValue = here->B4SOIcbsb;
            value->rValue *= here->B4SOIm;
            return(OK);
        case B4SOI_VON:
            value->rValue = here->B4SOIvon; 
            value->rValue *= here->B4SOIm;
            return(OK);
        case B4SOI_VDSAT:
            value->rValue = here->B4SOIvdsat; 
            value->rValue *= here->B4SOIm;
            return(OK);
        case B4SOI_QBS:
            value->rValue = *(ckt->CKTstate0 + here->B4SOIqbs); 
            return(OK);
        case B4SOI_QBD:
            value->rValue = *(ckt->CKTstate0 + here->B4SOIqbd); 
            return(OK);
#ifdef B4SOI_DEBUG_OUT
        case B4SOI_DEBUG1:
            value->rValue = here->B4SOIdebug1; 
            return(OK);
        case B4SOI_DEBUG2:
            value->rValue = here->B4SOIdebug2; 
            return(OK);
        case B4SOI_DEBUG3:
            value->rValue = here->B4SOIdebug3; 
            return(OK);
#endif
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}

