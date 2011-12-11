/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine sets instance parameters for NBJT2s in the circuit.
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "nbjt2def.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
NBJT2param(int param, IFvalue *value, GENinstance *inInst, IFvalue *select)
{
  register NBJT2instance *inst = (NBJT2instance *) inInst;

  NG_IGNORE(select);

  switch (param) {
  case NBJT2_WIDTH:
    inst->NBJT2width = value->rValue;
    inst->NBJT2widthGiven = TRUE;
    break;
  case NBJT2_AREA:
    inst->NBJT2area = value->rValue;
    inst->NBJT2areaGiven = TRUE;
    break;
  case NBJT2_OFF:
    inst->NBJT2off = TRUE;
    break;
  case NBJT2_IC_FILE:
    inst->NBJT2icFile = value->sValue;
    inst->NBJT2icFileGiven = TRUE;
    break;
  case NBJT2_PRINT:
    inst->NBJT2print = value->iValue;
    inst->NBJT2printGiven = TRUE;
    break;
  case NBJT2_TEMP:
    inst->NBJT2temp = value->rValue + CONSTCtoK;
    inst->NBJT2tempGiven = TRUE;
    break;
  default:
    return (E_BADPARM);
  }
  return (OK);
}
