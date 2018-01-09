/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1990 Jaijeet S. Roychowdhury
**********/

/*
 * This routine gives access to the internal device parameter of LTRA lines
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "ltradefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* ARGSUSED */
int
LTRAask(CKTcircuit *ckt, GENinstance *inst, int which, IFvalue *value, IFvalue *select)
{
  LTRAinstance *here = (LTRAinstance *) inst;
  int temp;

  NG_IGNORE(select);
  NG_IGNORE(ckt);

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
    value->rValue = LTRAmodPtr(here)->LTRAimped;
    return (OK);
  case LTRA_MOD_TD:
    value->rValue = LTRAmodPtr(here)->LTRAtd;
    return (OK);
  case LTRA_MOD_NL:
    value->rValue = LTRAmodPtr(here)->LTRAnl;
    return (OK);
  case LTRA_MOD_FREQ:
    value->rValue = LTRAmodPtr(here)->LTRAf;
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
    value->rValue = LTRAmodPtr(here)->LTRAreltol;
    return (OK);
  case LTRA_MOD_ABSTOL:
    value->rValue = LTRAmodPtr(here)->LTRAabstol;
    return (OK);
  case LTRA_BR_EQ1:
    value->rValue = here->LTRAbrEq1;
    return (OK);
  case LTRA_BR_EQ2:
    value->rValue = here->LTRAbrEq2;
    return (OK);
  case LTRA_DELAY:
    /*
     * value->v.vec.rVec = TMALLOC(double, here->LTRAsizeDelay);
     * value->v.numValue = temp = here->LTRAsizeDelay; while (temp--) {
     * value->v.vec.rVec++ = *here->LTRAdelays++;
     */
    value->v.vec.rVec = NULL;
    value->v.numValue = temp = 0;
    return (OK);
  default:
    return (E_BADPARM);
  }
  /* NOTREACHED */
}
