/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1990 Jaijeet S. Roychowdhury
**********/

/*
 * This routine gives access to the internal device parameter of LTRA lines
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "ifsim.h"
#include "ltradefs.h"
#include "sperror.h"
#include "suffix.h"

/* ARGSUSED */
int
LTRAask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, IFvalue *select)
{
  LTRAinstance *here = (LTRAinstance *) inst;
  int temp;

  switch (which) {
  case LTRA_POS_NODE1:
    value->iValue = here->LTRAposNode1;
    return (OK);
  case LTRA_NEG_NODE1:
    value->iValue = here->LTRAnegNode1;
    return (OK);
  case LTRA_POS_NODE2:
    value->iValue = here->LTRAposNode2;
    return (OK);
  case LTRA_NEG_NODE2:
    value->iValue = here->LTRAnegNode2;
    return (OK);
  case LTRA_MOD_Z0:
    value->rValue = here->LTRAmodPtr->LTRAimped;
    return (OK);
  case LTRA_MOD_TD:
    value->rValue = here->LTRAmodPtr->LTRAtd;
    return (OK);
  case LTRA_MOD_NL:
    value->rValue = here->LTRAmodPtr->LTRAnl;
    return (OK);
  case LTRA_MOD_FREQ:
    value->rValue = here->LTRAmodPtr->LTRAf;
    return (OK);
  case LTRA_V1:
    value->rValue = here->LTRAinitVolt1;
    return (OK);
  case LTRA_I1:
    value->rValue = here->LTRAinitCur1;
    return (OK);
  case LTRA_V2:
    value->rValue = here->LTRAinitVolt2;
    return (OK);
  case LTRA_I2:
    value->rValue = here->LTRAinitCur2;
    return (OK);
  case LTRA_MOD_RELTOL:
    value->rValue = here->LTRAmodPtr->LTRAreltol;
    return (OK);
  case LTRA_MOD_ABSTOL:
    value->rValue = here->LTRAmodPtr->LTRAabstol;
    return (OK);
  case LTRA_BR_EQ1:
    value->rValue = here->LTRAbrEq1;
    return (OK);
  case LTRA_BR_EQ2:
    value->rValue = here->LTRAbrEq2;
    return (OK);
  case LTRA_DELAY:
    /*
     * value->v.vec.rVec = (double *) MALLOC(here->LTRAsizeDelay);
     * value->v.numValue = temp = here->LTRAsizeDelay; while (temp--) {
     * value->v.vec.rVec++ = *here->LTRAdelays++;
     */
    value->v.vec.rVec = (double *) NULL;
    value->v.numValue = temp = 0;
    return (OK);
  default:
    return (E_BADPARM);
  }
  /* NOTREACHED */
}
