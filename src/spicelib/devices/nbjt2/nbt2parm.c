/**********
Copyright 1992 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
**********/

/*
 * This routine sets instance parameters for NBJT2s in the circuit.
 */

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "nbjt2def.h"
#include "sperror.h"
#include "suffix.h"

int
NBJT2param(param, value, inInst, select)
  int param;
  IFvalue *value;
  GENinstance *inInst;
  IFvalue *select;
{
  register NBJT2instance *inst = (NBJT2instance *) inInst;
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
    inst->NBJT2print = value->rValue;
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
