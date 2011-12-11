/**********
 * Copyright 1990 Regents of the University of California. All rights reserved.
 * File: b3ask.c
 * Author: 1995 Min-Chie Jeng and Mansun Chan. 
 * Modified by Paolo Nenzi 2002
 **********/

/* 
 * Release Notes: 
 * BSIM3v1v3.1,   Released by yuhua  96/12/08
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "bsim3v1def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
BSIM3v1ask (CKTcircuit * ckt, GENinstance * inst, int which, IFvalue * value,
	    IFvalue * select)
{
  BSIM3v1instance *here = (BSIM3v1instance *) inst;

  NG_IGNORE(select);

  switch (which)
    {
    case BSIM3v1_L:
      value->rValue = here->BSIM3v1l;
      return(OK);
    case BSIM3v1_W:
      value->rValue = here->BSIM3v1w;
      return(OK);
    case BSIM3v1_M:
      value->rValue = here->BSIM3v1m;
      return(OK);
    case BSIM3v1_AS:
      value->rValue = here->BSIM3v1sourceArea;
      return(OK);
    case BSIM3v1_AD:
      value->rValue = here->BSIM3v1drainArea;
      return(OK);
    case BSIM3v1_PS:
      value->rValue = here->BSIM3v1sourcePerimeter;
      return(OK);
    case BSIM3v1_PD:
      value->rValue = here->BSIM3v1drainPerimeter;
      return(OK);
    case BSIM3v1_NRS:
      value->rValue = here->BSIM3v1sourceSquares;
      return(OK);
    case BSIM3v1_NRD:
      value->rValue = here->BSIM3v1drainSquares;
      return(OK);
    case BSIM3v1_OFF:
      value->rValue = here->BSIM3v1off;
      return(OK);
    case BSIM3v1_NQSMOD:
      value->iValue = here->BSIM3v1nqsMod;
      return(OK);
    case BSIM3v1_IC_VBS:
      value->rValue = here->BSIM3v1icVBS;
      return(OK);
    case BSIM3v1_IC_VDS:
      value->rValue = here->BSIM3v1icVDS;
      return(OK);
    case BSIM3v1_IC_VGS:
      value->rValue = here->BSIM3v1icVGS;
      return(OK);
    case BSIM3v1_DNODE:
      value->iValue = here->BSIM3v1dNode;
      return(OK);
    case BSIM3v1_GNODE:
      value->iValue = here->BSIM3v1gNode;
      return(OK);
    case BSIM3v1_SNODE:
      value->iValue = here->BSIM3v1sNode;
      return(OK);
    case BSIM3v1_BNODE:
      value->iValue = here->BSIM3v1bNode;
      return(OK);
    case BSIM3v1_DNODEPRIME:
      value->iValue = here->BSIM3v1dNodePrime;
      return(OK);
    case BSIM3v1_SNODEPRIME:
      value->iValue = here->BSIM3v1sNodePrime;
      return(OK);
    case BSIM3v1_SOURCECONDUCT:
      value->rValue = here->BSIM3v1sourceConductance;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_DRAINCONDUCT:
      value->rValue = here->BSIM3v1drainConductance;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_VBD:
      value->rValue = *(ckt->CKTstate0 + here->BSIM3v1vbd);
      return(OK);
    case BSIM3v1_VBS:
      value->rValue = *(ckt->CKTstate0 + here->BSIM3v1vbs);
      return(OK);
    case BSIM3v1_VGS:
      value->rValue = *(ckt->CKTstate0 + here->BSIM3v1vgs);
      return(OK);
    case BSIM3v1_VDS:
      value->rValue = *(ckt->CKTstate0 + here->BSIM3v1vds);
      return(OK);
    case BSIM3v1_CD:
      value->rValue = here->BSIM3v1cd;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_CBS:
      value->rValue = here->BSIM3v1cbs;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_CBD:
      value->rValue = here->BSIM3v1cbd;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_GM:
      value->rValue = here->BSIM3v1gm;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_GDS:
      value->rValue = here->BSIM3v1gds;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_GMBS:
      value->rValue = here->BSIM3v1gmbs;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_GBD:
      value->rValue = here->BSIM3v1gbd;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_GBS:
      value->rValue = here->BSIM3v1gbs;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_QB:
      value->rValue = *(ckt->CKTstate0 + here->BSIM3v1qb);
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_CQB:
      value->rValue = *(ckt->CKTstate0 + here->BSIM3v1cqb);
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_QG:
      value->rValue = *(ckt->CKTstate0 + here->BSIM3v1qg);
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_CQG:
      value->rValue = *(ckt->CKTstate0 + here->BSIM3v1cqg);
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_QD:
      value->rValue = *(ckt->CKTstate0 + here->BSIM3v1qd);
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_CQD:
      value->rValue = *(ckt->CKTstate0 + here->BSIM3v1cqd);
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_CGG:
      value->rValue = here->BSIM3v1cggb;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_CGD:
      value->rValue = here->BSIM3v1cgdb;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_CGS:
      value->rValue = here->BSIM3v1cgsb;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_CDG:
      value->rValue = here->BSIM3v1cdgb;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_CDD:
      value->rValue = here->BSIM3v1cddb;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_CDS:
      value->rValue = here->BSIM3v1cdsb;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_CBG:
      value->rValue = here->BSIM3v1cbgb;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_CBDB:
      value->rValue = here->BSIM3v1cbdb;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_CBSB:
      value->rValue = here->BSIM3v1cbsb;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_CAPBD:
      value->rValue = here->BSIM3v1capbd;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_CAPBS:
      value->rValue = here->BSIM3v1capbs;
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_VON:
      value->rValue = here->BSIM3v1von;
      return(OK);
    case BSIM3v1_VDSAT:
      value->rValue = here->BSIM3v1vdsat;
      return(OK);
    case BSIM3v1_QBS:
      value->rValue = *(ckt->CKTstate0 + here->BSIM3v1qbs);
      value->rValue *= here->BSIM3v1m;
      return(OK);
    case BSIM3v1_QBD:
      value->rValue = *(ckt->CKTstate0 + here->BSIM3v1qbd);
      value->rValue *= here->BSIM3v1m;
      return(OK);
    default:
      return (E_BADPARM);
    }
  /* NOTREACHED */
}
